import torch
import torch.nn.functional as F


def correlation(input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2):
    """Pure PyTorch reimplementation of the original CUDA correlation kernel.

    Faithful to correlation_cuda_kernel.cu (forward): for every displacement
    (dy, dx) in [-max_displacement, max_displacement] stepped by stride2,

        out[b, tc, y, x] = sum_{c, kernel window} input1 * input2_shifted
                           / (kernel_size**2 * C)

    with channel tc = (dy/stride2 + rad) * nd + (dx/stride2 + rad)
    (y-displacement is the outer/slow index, as in the CUDA kernel).
    Only stride1 == 1 is supported (the only configuration the models use).
    """
    assert stride1 == 1, "pure PyTorch fallback only supports stride1 == 1"
    b, c, h, w = input1.shape
    kernel_rad = (kernel_size - 1) // 2
    disp_rad = max_displacement // stride2
    nd = 2 * disp_rad + 1
    nelems = kernel_size * kernel_size * c

    r1 = F.pad(input1, (pad_size, pad_size, pad_size, pad_size))
    r2 = F.pad(input2, (pad_size, pad_size, pad_size, pad_size))

    channels = []
    for ty in range(-disp_rad, disp_rad + 1):
        for tx in range(-disp_rad, disp_rad + 1):
            acc = input1.new_zeros(b, h, w)
            for j in range(-kernel_rad, kernel_rad + 1):
                for i in range(-kernel_rad, kernel_rad + 1):
                    a = r1[:, :, pad_size + j: pad_size + j + h,
                             pad_size + i: pad_size + i + w]
                    bb = r2[:, :, pad_size + ty * stride2 + j: pad_size + ty * stride2 + j + h,
                              pad_size + tx * stride2 + i: pad_size + tx * stride2 + i + w]
                    acc = acc + (a * bb).sum(dim=1)
            channels.append(acc / nelems)

    return torch.stack(channels, dim=1)  # (B, nd*nd, H, W)


class Correlation(torch.nn.Module):
    """Pure PyTorch correlation for FlowNet2 (drop-in replacement for the CUDA op)."""
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        # kept for API compatibility; the original CUDA forward kernel does not use it either
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return correlation(input1, input2, self.pad_size, self.kernel_size,
                           self.max_displacement, self.stride1, self.stride2)


class CorrelationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        ctx.save_for_backward(input1, input2)
        ctx.params = (pad_size, kernel_size, max_displacement, stride1, stride2)
        return correlation(input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        pad_size, kernel_size, max_displacement, stride1, stride2 = ctx.params
        with torch.enable_grad():
            i1 = input1.detach().requires_grad_(True)
            i2 = input2.detach().requires_grad_(True)
            out = correlation(i1, i2, pad_size, kernel_size, max_displacement, stride1, stride2)
            grad_input1, grad_input2 = torch.autograd.grad(out, (i1, i2), grad_output.contiguous())
        return grad_input1, grad_input2, None, None, None, None, None, None
