import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models import FlowNet2
from utils.frame_utils import read_gen


def writeFlow(name, flow):
    with open(name, 'wb') as f:
        f.write(b'PIEH')
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


if __name__ == '__main__':
    args = type('Args', (), {'rgb_max': 255., 'fp16': False})()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FlowNet2(args).to(device)
    net.eval()

    checkpoint = torch.load('work/FlowNet2_checkpoint.pth.tar', map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    img1_path = 'work/chairs_data/links/0000000-img0.ppm'
    img2_path = 'work/chairs_data/links/0000000-img1.ppm'
    out_path = 'work/0000000-predicted.flo'

    pim1 = read_gen(img1_path)
    pim2 = read_gen(img2_path)
    images = np.array([pim1, pim2]).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        result = net(im).squeeze()

    data = result.cpu().numpy().transpose(1, 2, 0)
    writeFlow(out_path, data)
    print(f'Flow written to {out_path}')
