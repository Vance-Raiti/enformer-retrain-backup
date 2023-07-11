
'''
loading basenji data. Much of this file is inspired by:
https://github.com/lucidrains/enformer-pytorch/blob/main/evaluate_enformer_pytorch_correlation.ipynb
'''

import torch
import tensorflow as tf
import numpy as np
import os
import pyfaidx
import json

BASENJI_SEQUENCE_LENGTH = 131_072
ENFORMER_SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896

BASENJI_DIR = '/projectnb/aclab/datasets/basenji/barnyard/basenji_barnyard'

# basenji data uses these processed .ml.fa files instead of the
# standard fasta files.
FASTA_FILE = {
    'human': os.path.join(BASENJI_DIR, 'hg38.ml.fa'),
    'mouse': os.path.join(BASENJI_DIR, 'mm10.ml.fa'),

}

# tensorflow will try to eat all the GPU memory, leaving
# none for anyone else! We don't actually want to eat ANY gpu memory.
# So, let's tell tensorflow that there are actually not any gpus here.
tf.config.set_visible_devices([], 'GPU')

def load_sequence_bed(
    organism: str,
    split: str
):
    path = os.path.join(BASENJI_DIR, 'data', organism, 'sequences.bed')

    f_open = open(path, 'r')
    sequences = f_open.readlines()
    f_open.close()
    # with open(path, 'r') as f_open:
    #     sequences = f_open.readlines()

    sequences = [s.strip().split('\t') for s in sequences]
    sequences = [s for s in sequences if s[3] == split]


    return sequences


def get_metadata(organism: str, split: str):
    path = os.path.join(BASENJI_DIR, 'data', organism, 'statistics.json')
    with open(path) as f:
        return json.load(f)

def deserialize_tfr(serialized_example, metadata):

    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(
        target,
        (metadata['target_length'], metadata['num_targets'])
    )

    return {
        'sequence': sequence,
        'target': target
    }


def one_hot_to_str(one_hot):
    indices = np.argmax(one_hot, axis=-1)

    chars = np.array(['A', 'C', 'G', 'T', 'N'])

    return ''.join(chars[indices])



class BasenjiDataset(torch.utils.data.IterableDataset):
    '''
    Loads a the Basenji dataset as a pytorch Dataset object.
    '''

    def __init__(
        self,
        organism: str,
        split: str,
        seq_length: int=ENFORMER_SEQUENCE_LENGTH,
        num_resize_test: int=-1,
        fail_on_error_check: bool=True,
        num_threads: int=1,
    ):
        '''
        arguments:

        organism: which organism to load basenji data for ('human' or 'mouse')
        split: 'train', 'test' or 'valid'
        seq_length: how long the final sequences should be
        num_resize_test: do some extra error checking for the first num_resize_test
            iterations
        fail_on_error_check: if any of the error checking from above results in a problem,
            immediately exit. Otherwise, just count the problems.
        num_threads: how many threads the tf record reader should use.
        '''

        self.organism = organism
        self.split = split
        self.seq_length = seq_length
        self.num_resize_test = num_resize_test
        self.num_threads = num_threads

        self.sequences = load_sequence_bed(organism, split)

        self.fasta = pyfaidx.Fasta(FASTA_FILE[organism])

        self.fail_on_error_check = fail_on_error_check
        

        self.dataset = None
        self.length = None

        # all the complicated lambda at the end is doing is making sure that the
        # files with name split-N.tfr are sorted in order of increasing N.
        path = os.path.join(
                BASENJI_DIR, 'data', organism, 'tfrecords', f'{split}-*.tfr'
        )
        self.tfr_files = sorted(
            tf.io.gfile.glob(path),
            key=lambda x: int(x.split('-')[-1].split('.')[0])
        )



    def extract_sequence(self, chrom: str, old_start: int, old_stop:int):
        length = len(self.fasta[chrom])


        new_start = old_start - (self.seq_length - (old_stop - old_start)) // 2
        left_padding = 0 if new_start >=0 else -new_start
        new_start = max(new_start , 0)

        new_stop = new_start + self.seq_length
        right_padding = 0 if new_stop < length else new_stop - length + 1
        new_stop = min(new_stop, length-1)

        offset_to_original = left_padding + old_start - new_start


        sequence = 'N' * left_padding + str(self.fasta[chrom][new_start:new_stop]).upper() + 'N' * right_padding

        return sequence, new_start, new_stop, offset_to_original

    def __len__(self):
        if self.length is not None:
            return self.length

        self.length = 0
        path = os.path.join(BASENJI_DIR, 'data', self.organism, 'sequences.bed')
        with open(path) as file:
            for line in file:
                if self.split in line:
                    self.length += 1            
        return self.length
    def init_dataset(self):
        metadata = get_metadata(self.organism, self.split)
        def deserialize(tfr):
            return deserialize_tfr(tfr, metadata)
        
        self.dataset = tf.data.TFRecordDataset(
            self.tfr_files,
            compression_type='ZLIB',
            num_parallel_reads=self.num_threads
        ).map(deserialize)

           
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        assert worker_info is None, "Only support single process loading"
        # TODO: split the bed file up so that there is a separate bed file
        # for each tfrecord shard, which will enable us to do multi-worker loading
        # by just assigning a subset of shards to each worker.

        
        
        if self.dataset is None:
            self.init_dataset()
        
        dataset = self.dataset

        error_count = 0
        for i, tfr in enumerate(dataset):
            chrom, old_start, old_stop, split = self.sequences[i]
            old_start = int(old_start)
            old_stop = int(old_stop)

            new_sequence, start, stop, offset = self.extract_sequence(chrom, old_start, old_stop)

            if i < self.num_resize_test:
                old_sequence = one_hot_to_str(tfr['sequence'])
                if len(old_sequence) <= len(new_sequence):
                    if old_sequence != new_sequence[offset:offset + len(old_sequence)]:
                        error_count += 1
                else:
                    if new_sequence != old_sequence[-offset:-offset + len(new_sequence)]:
                        error_count += 1

            if self.fail_on_error_check and error_count > 0:
                raise AssertionError(f"sequence mismatch!\noriginal basenji sequence:\n{old_sequence}\nnew sequence:\n{new_sequence[offset:offset+len(old_sequence)]}")
            # perform one-hot encoding s.t. 'N' is uniform
            b = torch.Tensor(memoryview(new_sequence.encode('ascii')))
            x = torch.zeros((b.shape[0],4))
            n = (b==ord('N'))*0.25
            for i,c in enumerate(['A','C','T','G']):
                x[:,i]=(b==ord(c))+n
            yield {
                'features': x,
                #there is no way to convert directly from TF to torch from what I can tell
                'targets': torch.from_numpy(tfr['target'].numpy()),
                'chrom': chrom,
                'start': start,
                'stop': stop
            }








