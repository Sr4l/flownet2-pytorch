import torch


class ChannelNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        input1_c = input1.detach().clone()
        b, c, h, w = input1_c.size()
        norm = torch.norm(input1_c.view(b, c, -1), p=norm_deg, dim=1, keepdim=True)
        norm = norm.view(b, 1, h, w)
        ctx.save_for_backward(input1_c)
        ctx.norm_deg = norm_deg
        return norm

    @staticmethod
    def backward(ctx, grad_output):
        input1_c, = ctx.saved_tensors
        norm_deg = ctx.norm_deg
        b, c, h, w = input1_c.size()
        norm = torch.norm(input1_c.view(b, c, -1), p=norm_deg, dim=1, keepdim=True)
        norm_expanded = norm.view(b, 1, h, w).expand_as(input1_c)
        eps = 1e-9
        safe_norm = torch.clamp(norm_expanded, min=eps)
        normalized = input1_c / safe_norm
        grad_input1 = grad_output * normalized / safe_norm
        return grad_input1, None


class ChannelNorm(torch.nn.Module):
    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)