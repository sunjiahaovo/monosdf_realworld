import torch
import numpy as np
from training.common import *

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        self.weight = [1.]

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
        self.start, self.end = self.kwargs['barf_c2f']

    def embed(self, inputs, progress):
        start, end = self.start, self.end
        alpha = (progress-start)/(end-start)*self.kwargs['num_freqs']
        k = torch.arange(self.kwargs['num_freqs'], dtype=torch.float32)
        weight_ = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        
        for idx in range(self.kwargs['num_freqs']):
            for jdx in range(len(self.kwargs['periodic_fns'])):
                self.weight.append(weight_[idx])
        self.weight = torch.tensor(self.weight)
        
        res = []
        for idx, fn in enumerate(self.embed_fns):
            res_ = fn(inputs)
            weight_ = self.weight[idx]
            res.append(weight_*res_)
        res = torch.cat(res, -1)
        # self.weight = weight
        embed_weight_cal = self.weight
        print(embed_weight_cal)
        return res
        # return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        
    def weight_out(self):
        return self.weight

def get_embedder(multires, progress, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'barf_c2f': [0.1, 0.5],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # def embed(x, progress, eo=embedder_obj): return eo.embed(x, progress)
    embed = lambda x, prog = progress, eo=embedder_obj: eo.embed(x, prog)
    return embed, embedder_obj.out_dim, embedder_obj.weight_out()
