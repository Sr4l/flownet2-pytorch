import torch
import torch.nn.functional as F


def resample2d(input1, input2, bilinear=True):
    """Pure PyTorch reimplementation of the original Resample2d CUDA kernel.

    Warps input1 (B, C, H, W) by the per-pixel displacement field input2
    (B, 2, H, W) with (dx, dy) in pixel coordinates: out[b, c, y, x] samples
    input1 at (x + dx, y + dy). Out-of-bounds coordinates are clamped to the
    image border, as the CUDA kernel does. Only kernel_size == 1 is supported
    (the only configuration the models use).
    """
    b, _, h, w = input2.size()

    ys, xs = torch.meshgrid(
        torch.arange(h, device=input2.device, dtype=input2.dtype),
        torch.arange(w, device=input2.device, dtype=input2.dtype),
        indexing='ij')
    # sampling positions in pixel coordinates, (x, y) order in the last dim.
    # NOTE: the '+' allocates a new tensor; input2 is never modified in place.
    grid = torch.stack((xs, ys), dim=-1).unsqueeze(0) + input2.permute(0, 2, 3, 1)

    # normalize to [-1, 1] for align_corners=True
    grid = torch.stack(
        (2.0 * grid[..., 0] / max(w - 1, 1) - 1.0,
         2.0 * grid[..., 1] / max(h - 1, 1) - 1.0), dim=-1)

    return F.grid_sample(input1, grid.to(input1.dtype),
                         mode='bilinear' if bilinear else 'nearest',
                         padding_mode='border', align_corners=True)


class Resample2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, bilinear=True):
        assert kernel_size == 1, "pure PyTorch fallback only supports kernel_size == 1"
        input1_c = input1.contiguous()
        input2_c = input2.contiguous()
        ctx.save_for_backward(input1_c, input2_c)
        ctx.bilinear = bilinear
        return resample2d(input1_c, input2_c, bilinear)

    @staticmethod
    def backward(ctx, grad_output):
        input1_c, input2_c = ctx.saved_tensors
        with torch.enable_grad():
            i1 = input1_c.detach().requires_grad_(True)
            i2 = input2_c.detach().requires_grad_(True)
            out = resample2d(i1, i2, ctx.bilinear)
            grad_input1, grad_input2 = torch.autograd.grad(out, (i1, i2), grad_output.contiguous())
        return grad_input1, grad_input2, None, None


class Resample2d(torch.nn.Module):
    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        assert kernel_size == 1, "pure PyTorch fallback only supports kernel_size == 1"
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        input2_c = input2.contiguous()
        # plain differentiable ops: autograd computes correct grads for both inputs
        return resample2d(input1_c, input2_c, self.bilinear)
