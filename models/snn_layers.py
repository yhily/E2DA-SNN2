import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# SNN global hyper-parameters —— 单一来源 (single source of truth)
# 旧的三个模块各自定义 time_window (snn_layers=2, modules=2, model=4) 且互不相同，
# 会导致 Model.forward 平铺 4 个时间步而 SNN 层只消费前 2 个。此处统一为一份常量，
# 各层实际时间步数改为从输入张量的第 0 维动态推导，彻底消除不一致。
# -----------------------------------------------------------------------------
THRESH = 0.5  # 神经元发放阈值 (neuronal threshold)
LENS = 0.5  # 代理梯度宽度 (hyper-parameters of approximate function)
DECAY = 0.25  # 膜电位衰减常数 (decay constants)
TIME_WINDOW = 4  # 默认时间窗：Model.forward 将输入在时间维平铺的份数

# 向后兼容别名（原有全局名，供外部代码引用）
thresh = THRESH
lens = LENS
decay = DECAY
time_window = TIME_WINDOW


# LIF 激活：阶跃函数 + 盒状代理梯度
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(THRESH).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - THRESH) < LENS
        temp = temp / (2 * LENS)
        return grad_input * temp.float()


act_fun = ActFun.apply


class mem_update(nn.Module):
    """LIF 膜电位更新。输入/输出均为 (T, B, C, H, W)。

    时间步数 T 从输入第 0 维动态获取，不再依赖全局常量。
    """

    def __init__(self, act=False):
        super(mem_update, self).__init__()
        self.actFun = nn.SiLU()
        self.act = act

    def forward(self, x):
        T = x.shape[0]
        mem = torch.zeros_like(x[0])
        spike = torch.zeros_like(x[0])
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(T):
            if i >= 1:
                mem = mem_old * DECAY * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            if self.act:
                spike = self.actFun(mem)
            else:
                spike = act_fun(mem)

            mem_old = mem.clone()
            output[i] = spike
        return output


class Snn_Conv2d(nn.Conv2d):
    """时间维卷积：输入 (T, B, C, H, W) -> 输出 (T, B, C_out, H_out, W_out)。

    将 (T, B) 合并为一批做单次 conv2d，再还原形状，等价于逐时间步卷积但快得多。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        T, B = input.shape[0], input.shape[1]
        x = input.reshape(T * B, *input.shape[2:])
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y.view(T, B, *y.shape[1:])


class batch_norm_2d(nn.Module):
    """Spiking 时空 BatchNorm：把 (T, B, C, H, W) 视作 (N, C, D, H, W)，在 (N, D, H, W) 上归一化。"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)  # input (N,C,D,H,W) spatio-temporal Batch Normalization

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):  # 5
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, THRESH)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2 * THRESH)
            nn.init.zeros_(self.bias)
