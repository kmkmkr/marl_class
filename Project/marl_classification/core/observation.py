# -*- coding: utf-8 -*-
import operator as op
from functools import reduce

import torch as th

def obs_generic_patch(patched_img: th.Tensor, pos: th.Tensor, ) -> th.Tensor:
    '''
    patched_img: (n_agents, b, hidden_dim, patch_size, patch_size)
    pos: (n_agents, b, 2)
    output: (n_agents, b, hidden_dim)
    '''
    if len(patched_img.size()) == 4: # (b, hidden_dim, patch_size, patch_size)
        patched_img = patched_img[None,...].repeat(pos.size(0), 1, 1, 1, 1)
    # pos0: (n_agents, b, 2) -> (n_agents, b, 1, 1, 2)  -> (n_agents, b, hidden, 1, patch_size, hidden_dim)
    # pos1: (n_agents, b, 2) -> (n_agents, b, 1, 1, 2)  -> (n_agents, b, hidden, 1, 1, hidden_dim)
    pos0 = pos[:, :, 0, None, None,None].repeat(1,1,1,patched_img.size(-1),patched_img.size(2))
    pos1 = pos[:, :, 1, None, None,None].repeat(1,1,1,1,patched_img.size(2))
    # (n_agents, b, patch_size, patch_size, hidden_dim)
    print(patched_img.size(), pos0.size(), pos1.size())
    patched_img = patched_img.permute(0, 1, 3, 4, 2)
    print(patched_img.size(), pos0.size(), pos1.size())
    print(pos0.min(), pos0.max(), pos1.min(), pos1.max())
    print(set(pos0.view(-1).tolist()), set(pos1.view(-1).tolist()))
    return patched_img.gather(2, pos0).gather(3, pos1)[:,:,0,0,:]

def obs_generic(x: th.Tensor, pos: th.Tensor, f: int, expand_x = None,return_mask=False) -> th.Tensor:
    '''
    x: (b, c, h, w)
    pos: (n_agents, b, 2)
    f: window size
    output: (n_agents, b, c, f, f)
    '''
    x_sizes = x.size()
    b_img, c = x_sizes[0], x_sizes[1]
    sizes = list(x_sizes[2:])

    nb_a, _, _ = pos.size()

    pos_min = pos
    pos_max = pos_min + f

    masks = []

    for d, s in enumerate(sizes):
        values = th.arange(0, s, device=pos.device)

        mask = (pos_min[:, :, d, None] <= values.view(1, 1, s)) & (
            values.view(1, 1, s) < pos_max[:, :, d, None]
        )

        for n_unsq in range(len(sizes) - 1):
            mask = mask.unsqueeze(-2) if n_unsq < d else mask.unsqueeze(-1)

        masks.append(mask)
    mask = reduce(op.and_, masks)
    mask = mask.unsqueeze(2)
    if expand_x is not None:
        x = expand_x
    else:
        x = x.unsqueeze(0)
    if return_mask:
        if expand_x is not None:
            return (
                x.masked_select(mask)
                .view(nb_a, b_img, c, *[f for _ in range(len(sizes))])
        ), mask
        return  (
                x.masked_select(mask)
                .view(nb_a, b_img, c, *[f for _ in range(len(sizes))])
        ), mask
    return (
        x.masked_select(mask)
        .view(nb_a, b_img, c, *[f for _ in range(len(sizes))])
    )
