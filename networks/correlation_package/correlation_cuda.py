import torch
import torch.nn.functional as F


class Correlation(torch.nn.Module):
    """Pure PyTorch correlation for FlowNet2."""
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        b, c, h, w = input1.shape
        d = self.max_displacement
        s2 = self.stride2
        k = self.kernel_size

        # Pad inputs
        in1 = F.pad(input1, (d, d, d, d))  # (B, C, H+2d, W+2d)
        in2 = F.pad(input2, (d, d, d, d))

        # Number of displacement offsets in each axis
        nd = 2 * (d // s2) + 1

        # Output: (B, C, h, w, nd, nd)
        out = torch.zeros(b, c, h, w, nd, nd, device=input1.device, dtype=input1.dtype)

        # For each offset, compute correlation
        for i in range(nd):
            di = i - d // s2
            for j in range(nd):
                dj = j - d // s2
                # Shifted input2 at (di, dj)
                si = d + di
                sj = d + dj
                in2_shifted = in2[:, :, si:si + h, sj:sj + w]
                out[:, :, :, :, i, j] = in1[:, :, d:d + h, d:d + w] * in2_shifted

        # Sum over channels and spatial kernel
        out = out.sum(dim=1)  # (B, h, w, nd, nd)

        return out.permute(0, 3, 4, 1, 2).reshape(b, nd * nd, h, w)


class CorrelationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        return Correlation(
            pad_size=pad_size, kernel_size=kernel_size,
            max_displacement=max_displacement, stride1=stride1,
            stride2=stride2, corr_multiply=corr_multiply
        )(input1, input2)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None, None