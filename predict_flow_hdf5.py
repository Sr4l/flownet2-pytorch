import os
import sys

import h5py
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models import FlowNet2


def main():
    hdf5_path = '/daten/Work/kistner/inter_velo.h5'
    dataset_path = '/m0/q007_0/diff_bg'
    checkpoint_path = 'work_pretrained/FlowNet2_JR_CD_fintune_train-checkpoint_1500ep.pth.tar'

    with h5py.File(hdf5_path, 'r') as f:
        frames = f[dataset_path][:]  # (N, H, W)

    N, orig_H, orig_W = frames.shape
    print(f'Loaded {N} frames of shape ({orig_H}, {orig_W})')

    # Track NaN mask per frame (original dims)
    nan_mask = np.isnan(frames)  # (N, orig_H, orig_W)
    print(f'NaN fraction: {nan_mask.sum() / frames.size:.3%}')
    frames = np.nan_to_num(frames, nan=0.0)

    # Normalize to [0, 255] (data is float, not uint8)
    lo, hi = frames.min(), frames.max()
    if hi > lo:
        frames = (frames - lo) / (hi - lo) * 255.0
    else:
        frames[:] = 0
    print(f'Normalized data range [{lo:.4f}, {hi:.4f}] → [0, 255]')

    # Grayscale → repeat to 3 channels
    frames = np.stack([frames] * 3, axis=-1).astype(np.float32)

    # Pad H, W to multiples of 64
    pad_h = (64 - orig_H % 64) % 64
    pad_w = (64 - orig_W % 64) % 64
    if pad_h or pad_w:
        frames = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        print(f'Padded to ({orig_H + pad_h}, {orig_W + pad_w})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = type('Args', (), {'rgb_max': 255., 'fp16': False})()
    net = FlowNet2(args).to(device)
    net.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    H, W = frames.shape[1], frames.shape[2]
    num_pairs = N - 1
    flow_array = np.zeros((num_pairs, H, W, 2), dtype=np.float32)

    with torch.no_grad():
        for i in range(num_pairs):
            img1 = frames[i]
            img2 = frames[i + 1]
            im = np.array([img1, img2]).transpose(3, 0, 1, 2)  # (3, 2, H, W)
            im = torch.from_numpy(im).unsqueeze(0).to(device)  # (1, 3, 2, H, W)

            result = net(im)[0].squeeze(0)  # (2, H, W)
            flow_array[i] = result.cpu().numpy().transpose(1, 2, 0)

            if (i + 1) % 100 == 0 or i == 0:
                print(f'  {i+1}/{num_pairs} pairs processed')

    # Crop back to original spatial dimensions
    flow_array = flow_array[:, :orig_H, :orig_W, :]

    # Restore NaN: if either frame in a pair had NaN, set output to NaN
    pair_nan = nan_mask[:num_pairs] | nan_mask[1:]  # (num_pairs, orig_H, orig_W)
    flow_array[pair_nan] = np.nan  # broadcasts over last (channel) axis
    nan_fraction = pair_nan.sum() / pair_nan.size
    print(f'Output NaN fraction: {nan_fraction:.3%}')

    out_path = '/daten/Work/kistner/inter_velo_flow.h5'
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('diff_bg_flow', data=flow_array, compression='gzip')
    print(f'Flow saved to {out_path} — shape {flow_array.shape}, dtype {flow_array.dtype}')


if __name__ == '__main__':
    main()