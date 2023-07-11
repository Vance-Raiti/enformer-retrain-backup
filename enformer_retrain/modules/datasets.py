from .basenji import BasenjiDataset
import torch

class RandM(torch.utils.data.Dataset):
    '''
        random matrix

        dataset that always returns a square matrix of length sequence_length.
        Made for testing the pipeline in conjunction with 'linear' architecture
    '''
    def __init__(self,**kwargs):
        self.dim = 4

    def __len__(self):
        return int(3e4)

    def __getitem__(self,idx):
        return {
            'features': torch.rand((self.dim,self.dim)),
            'targets': torch.rand((self.dim,self.dim)),
        }


class TinyBasenji(BasenjiDataset):
    def __len__(self):
        return 50
    def __iter__(self):
        for i,x in enumerate(super()):
            if i > len(self):
                break
            yield x
            

Datasets = {
    'basenji': BasenjiDataset,
    'randm': RandM,
    # I'm sure there's a better way to do this
    'tiny-basenji' : TinyBasenji,
}
