import torch, cv2
import torch.nn as nn
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_visualize_attention(vit, img, patch_size, device):
    if len(img.size()) == 3:
        img = img.unsqueeze(0)
    b, c, h, w = img.size()
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    
    attn = vit.get_last_selfattention(img.to(device)).cpu()
    nh = attn.shape[1] # number of head
    
    # we keep only the output patch attention
    attn = attn[:, :, 0, 1:].reshape(b, nh, -1)
    
    attn = attn.reshape(b, nh, w_featmap, h_featmap)
    
    # (b, nh, w_f, h_f) -> (b, 1, w_f, h_f)
    attn = attn.mean(1).unsqueeze(1)
    attn = nn.functional.interpolate(attn, scale_factor=patch_size, mode="nearest")
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    img = (img - img.min()) / (img.max() - img.min())
    return attn.squeeze(0), img.squeeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam