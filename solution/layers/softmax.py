import torch
from torch import nn

class SoftmaxLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a softmax calculation.
        Given a matrix x (n, d) on each element performs:

        softmax(x) = exp(x_ij) / sum_k=0^d exp(x_ik)

        i.e. it first takes an exponential of each element,
            and that normalizes rows so that their L-1 norm is equal to 1.

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape (n, d).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, also with shape (n, d).
                Each row has L-1 norm of 1, and each element is in [0, 1] (i.e. each row is a probability vector).
                Output data.
        """
        x_stable =  x - torch.max(x)
        exp_x = torch.exp(x_stable)
        sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
        return exp_x / sum_exp_x
