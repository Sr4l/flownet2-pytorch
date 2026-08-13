import torch
import torch.nn.functional as F

class Resample2dModule(torch.nn.Module):
    """Pure PyTorch implementation of Resample2d using grid_sample."""

    def forward(self, input1, input2):
        """
        input1: (B, C, H, W) - image to warp
        input2: (B, 2, H, W) - displacement field (dx, dy) in pixel coordinates
        """
        b, _, h, w = input2.size()

        x, y = torch.meshgrid(torch.arange(w, device=input1.device),
                              torch.arange(h, device=input1.device),
                              indexing='xy')
        grid = torch.stack([x.float(), y.float()], dim=-1)

        grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(w - 1, 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(h - 1, 1) - 1.0

        flow = input2.permute(0, 2, 3, 1)
        flow[:, :, :, 0] = flow[:, :, :, 0] * 2.0 / max(w - 1, 1)
        flow[:, :, :, 1] = flow[:, :, :, 1] * 2.0 / max(h - 1, 1)

        grid = grid.unsqueeze(0).expand(b, -1, -1, -1) + flow

        output = F.grid_sample(input1, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return output


class Resample2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, bilinear=True):
        input1_c = input1.contiguous()
        input2_c = input2.contiguous()
        b, _, h, w = input2_c.size()

        x, y = torch.meshgrid(torch.arange(w, device=input1_c.device),
                              torch.arange(h, device=input1_c.device),
                              indexing='xy')
        grid = torch.stack([x.float(), y.float()], dim=-1)
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(w - 1, 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(h - 1, 1) - 1.0

        flow = input2_c.permute(0, 2, 3, 1)
        flow[:, :, :, 0] = flow[:, :, :, 0] * 2.0 / max(w - 1, 1)
        flow[:, :, :, 1] = flow[:, :, :, 1] * 2.0 / max(h - 1, 1)

        grid = grid.unsqueeze(0).expand(b, -1, -1, -1) + flow
        ctx.save_for_backward(input1_c, grid)
        return F.grid_sample(input1_c, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    @staticmethod
    def backward(ctx, grad_output):
        input1, grid = ctx.saved_tensors
        grad_input1 = F.grid_sample(grad_output, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return grad_input1, None, None, None


class Resample2d(torch.nn.Module):
    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        input2_c = input2.contiguous()
        return Resample2dFunction.apply(input1_c, input2_c, self.kernel_size, self.bilinear)