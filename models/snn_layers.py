"""Core spiking layers used by E2DA-SNN.

All layers in this module receive and return tensors arranged as
``[time, batch, channels, height, width]``. Static RGB images are expanded
along the leading time dimension by :class:`models.model.Model`.

The lower-case aliases at the end of this file are intentionally retained so
that existing YAML files and serialized checkpoints remain loadable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SPIKE_THRESHOLD = 0.5
SURROGATE_WINDOW = 0.5
MEMBRANE_DECAY = 0.25
TIME_STEPS = 2


class SurrogateSpike(torch.autograd.Function):
    """Binary spike function with a rectangular surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane):
        ctx.save_for_backward(membrane)
        return membrane.gt(SPIKE_THRESHOLD).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membrane,) = ctx.saved_tensors
        within_window = (membrane - SPIKE_THRESHOLD).abs() < SURROGATE_WINDOW
        surrogate = within_window.float() / (2 * SURROGATE_WINDOW)
        return grad_output * surrogate


surrogate_spike = SurrogateSpike.apply


class LIFActivation(nn.Module):
    """Apply leaky integrate-and-fire dynamics across the time dimension.

    Args:
        analog: If ``True``, use SiLU activations instead of binary spikes.
            This compatibility option is preserved from the original code.
    """

    def __init__(self, analog=False, act=None):
        super().__init__()
        if act is not None:  # legacy keyword used by the original implementation
            analog = act
        self.actFun = nn.SiLU()  # legacy attribute retained for serialized checkpoints
        self.act = analog

    def forward(self, inputs):
        """Return activations with the same ``[T, B, C, H, W]`` shape."""
        membrane = torch.zeros_like(inputs[0])
        spike = torch.zeros_like(inputs[0])
        outputs = torch.zeros_like(inputs)

        for step in range(inputs.shape[0]):
            if step == 0:
                membrane = inputs[step]
            else:
                membrane = membrane * MEMBRANE_DECAY * (1 - spike.detach()) + inputs[step]

            spike = self.actFun(membrane) if self.act else surrogate_spike(membrane)
            outputs[step] = spike

        return outputs


class SpikingConv2d(nn.Conv2d):
    """Apply one shared 2-D convolution independently at every time step."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        marker="b",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.marker = marker

    def forward(self, inputs):
        """Convolve a ``[T, B, C, H, W]`` spike tensor."""
        return torch.stack(
            [
                F.conv2d(
                    inputs[step],
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                for step in range(inputs.shape[0])
            ],
            dim=0,
        )


class ThresholdBatchNorm3d(nn.BatchNorm3d):
    """Temporal batch normalization initialized at the firing threshold."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, SPIKE_THRESHOLD)
            nn.init.zeros_(self.bias)


class ResidualBatchNorm3d(nn.BatchNorm3d):
    """Lower-gain temporal batch normalization used at residual outputs."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2 * SPIKE_THRESHOLD)
            nn.init.zeros_(self.bias)


class TemporalBatchNorm(nn.Module):
    """Batch-normalize spike tensors while preserving ``[T, B, C, H, W]``."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = ThresholdBatchNorm3d(num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        batch_first = inputs.permute(1, 2, 0, 3, 4).contiguous()
        normalized = self.bn(batch_first)
        return normalized.permute(2, 0, 1, 3, 4).contiguous()


class ResidualTemporalBatchNorm(nn.Module):
    """Temporal batch normalization with lower residual-branch gain."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = ResidualBatchNorm3d(num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        batch_first = inputs.permute(1, 2, 0, 3, 4).contiguous()
        normalized = self.bn(batch_first)
        return normalized.permute(2, 0, 1, 3, 4).contiguous()


# Backward-compatible names used by existing model YAML files and checkpoints.
thresh = SPIKE_THRESHOLD
lens = SURROGATE_WINDOW
decay = MEMBRANE_DECAY
time_window = TIME_STEPS
ActFun = SurrogateSpike
act_fun = surrogate_spike
mem_update = LIFActivation
Snn_Conv2d = SpikingConv2d
BatchNorm3d1 = ThresholdBatchNorm3d
BatchNorm3d2 = ResidualBatchNorm3d
batch_norm_2d = TemporalBatchNorm
batch_norm_2d1 = ResidualTemporalBatchNorm
