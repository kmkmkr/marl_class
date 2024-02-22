import torch as th
import torch.nn as nn
from torchvision import transforms as pth_transforms
from .observation import obs_generic, obs_generic_patch
import time



def reward_func(pos, img, model, attn=None,image_size=224, patch_size=8, window_size=2, device="cuda", is_patch=False):
    """
    pos: (n_step, n_agents, b, 2)
    img: (b, c, h, w)
    attn: (b, 1, w_featmap, h_featmap)
    """
    ns, na, nb, _ = pos.size()
    attn_returns = th.zeros((ns, na, nb), device=th.device(device))
    if len(attn.size()) == 3:
        attn = attn.unsqueeze(1)
    if not is_patch:
        attn = nn.functional.interpolate(attn, scale_factor=patch_size, mode="nearest")
        tmp_attn = attn.expand(na, -1, -1, -1, -1).clone()
    for step_idx in range(ns):
        if attn is None:
            # img = transform(img).to(device)
            b, c, h, w = img.size()
            w_featmap = img.shape[-2] // patch_size
            h_featmap = img.shape[-1] // patch_size
            
            attn = model.get_last_selfattention(img.to(device)).cpu()
            nh = attn.shape[1] # number of head
            
            # we keep only the output patch attention
            attn = attn[:, :, 0, 1:].reshape(b, nh, -1)
            
            attn = attn.reshape(b, nh, w_featmap, h_featmap)
            
            # (b, nh, w_f, h_f) -> (b, 1, w_f, h_f)
            attn = attn.mean(1).unsqueeze(1)
        # (b, 1, w_featmap, h_featmap) -> (b, 1, h, w)
        if is_patch:
            attn_returns[step_idx] = obs_generic_patch(attn, pos[step_idx].cpu()).mean(-1).to(device)
        else:
            rewards, mask = obs_generic(attn, pos[step_idx].cpu(), window_size,return_mask=True, expand_x=tmp_attn)
            attn_returns[step_idx] = rewards.mean((2,3,4)).to(device)
            # print(mask.size(), attn.size(), rewards.size(), attn_returns[step_idx].size(), pos[step_idx].size(), pos.size())
            tmp_attn[mask] = 0
    return attn_returns