import argparse, pickle, sys, os
from tqdm.auto import tqdm
from utilities.tools import read_yamls
### python3.10 attrduct (https://github.com/wallento/wavedrompy/issues/32)
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdict import AttrDict
sys.path.append('/home/nkmur/lab/sl_rl/dino')
import torch as th
import torch.nn as nn
import torchvision.transforms as pth_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def run(conf):
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = th.hub.load(conf.repo_or_dir, conf.model_name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)
    transform = pth_transforms.Compose([
        pth_transforms.Resize((conf.image_size, conf.image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(conf.data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)
    if conf.is_patch:
        attn_array = th.zeros(len(dataset), conf.image_size//conf.patch_size, conf.image_size//conf.patch_size, dtype=th.float32)
    else:
        attn_array = th.zeros(len(dataset), conf.image_size, conf.image_size, dtype=th.float32)
    # img_array = th.zeros(len(dataset), 3, conf.image_size, conf.image_size, dtype=th.float32)
    # label_array = th.zeros(len(dataset), dtype=th.int16)
    with tqdm(data_loader, total=len(data_loader)) as pbar:
        for batch_idx, ( img, label ) in enumerate(pbar):
            # make the image divisible by the patch size
            # w, h = img.shape[1] - img.shape[1] % conf.patch_size, img.shape[2] - img.shape[2] % conf.patch_size
            # img = img[:, :, :w, :h]

            w_featmap = img.shape[-2] // conf.patch_size
            h_featmap = img.shape[-1] // conf.patch_size

            attentions = model.get_last_selfattention(img.to(device))
            nh = attentions.shape[1] # number of head

            # we keep only the output patch attention
            attentions = attentions[:, :, 0, 1:].reshape(conf.batch_size, nh, -1)
            attentions = attentions.reshape(img.shape[0], nh, w_featmap, h_featmap)
            if conf.threshold is not None and conf.threshold > 0:
                # we keep only a certain percentage of the mass
                val, idx = th.sort(attentions)
                val /= th.sum(val, dim=1, keepdim=True)
                cumval = th.cumsum(val, dim=1)
                th_attn = cumval > (1 - conf.threshold)
                idx2 = th.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            if not conf.is_patch:
                attentions = nn.functional.interpolate(attentions, scale_factor=conf.patch_size, mode="nearest")

            attn_array[batch_idx*conf.batch_size:(batch_idx+1)*conf.batch_size] = attentions.cpu().detach().mean(1)
            # img_array[batch_idx*conf.batch_size:(batch_idx+1)*conf.batch_size] = img.cpu()
            # label_array[batch_idx*conf.batch_size:(batch_idx+1)*conf.batch_size] = label
    os.makedirs(conf.save_path, exist_ok=True)
    with open(os.path.join(conf.save_path, "data.pkl"), 'wb') as f:
        pickle.dump({'attn': attn_array}, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args, remaining = parser.parse_known_args()
    
    configs = read_yamls('./config')
    conf = AttrDict(configs[args.config])
    run(conf)