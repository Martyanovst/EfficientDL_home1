import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        batch_shape, vector_shape = input.shape[0], input.shape[1]
        n = batch_shape * dist.get_world_size()
        ssum = torch.sum(input, dim=0)
        ssum_squared = torch.sum(torch.square(input), dim=0)
        sync_tensor = torch.stack((ssum, ssum_squared))
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
        ssum, ssum_squared = sync_tensor
        mean = ssum / n
        std = (ssum_squared / n) - (mean ** 2)
        running_mean = momentum * running_mean + mean * (1 - momentum)
        running_std = momentum * running_std + std * (1 - momentum)
        output = (input - mean) / torch.sqrt(std + eps)
        ctx.save_for_backward(output, torch.tensor([n], device=std.device), std, torch.tensor([eps]))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        output, n, output_std, eps = ctx.saved_tensors
        grad_output_sum = grad_output.sum(dim=0)
        output_grad_output_sum = torch.sum(output * grad_output, dim=0)
        sync_tensor = torch.stack((grad_output_sum, output_grad_output_sum))
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
        grad_output_sum, output_grad_output_sum = sync_tensor
        grad_input = grad_output * n
        grad_input -= grad_output_sum
        grad_input -= output * output_grad_output_sum
        grad_input /= n * torch.sqrt(output_std + eps)
        return grad_input, None, None, None, None

class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, device, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=device,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,), device=device)
        self.running_std = torch.ones((num_features,), device=device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        if self.training:
            return sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)
        else:
            return (input - self.running_mean) / torch.sqrt(self.running_std + self.eps)

