import torch
import wandb

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def select(name,kwargs,dictionary,objs=None):
    keywords = list(kwargs.keys())
    for kw in keywords:
        arg = kwargs[kw]
        if arg is None:
            del kwargs[kw]

    if name not in dictionary:
        raise ValueError(f"{name} is not a supported option in {dictionary['name']}\nSupported: {dictionary.keys()}")
    bp = {**kwargs,'name':name}
    if objs is not None:
        kwargs = {**objs,**kwargs}
    obj = dictionary[name](**kwargs)
    setattr(obj,'blueprint',bp)
    return obj

def ld(label,sd,dictionary,objs=None):
    bp = sd[label+'-bp']
    if objs is not None:
        kwargs = {**bp,**objs}
    else:
        kwargs = bp
    bp['name'] = kwargs.pop('name')
    obj = dictionary[bp['name']](**kwargs)
    setattr(obj,'blueprint',bp)
    obj.load_state_dict(sd[label])
    return obj

def checkpoint(obj,label,sd):
    if hasattr(obj,'to'):
        obj.to('cpu')
    sd[label]=obj.state_dict()
    sd[label+'-bp']=obj.blueprint
    assert 'name' in obj.blueprint
    if hasattr(obj,'to'):
        obj.to(device())

