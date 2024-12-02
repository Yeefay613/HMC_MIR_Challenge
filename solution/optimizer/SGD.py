import torch

class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float) -> None:
        """Constructor for Stochastic Gradient Descent (SGD) Optimizer.

        Provided code contains call to super class, which will initialize paramaters properly (see step function docs).
        This class will only update the parameters provided to it, based on their (already calculated) gradients.

        Args:
            params: Parameters to update each step. You don't need to do anything with them.
                They are properly initialize through the super call.
            lr (float): Learning Rate of the gradient descent.
        """
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):  # noqa: E251
        """
        Performs a step of gradient descent. You should loop through each parameter, and update it's value based on its gradient, value and learning rate.

        Args:
            closure (optional): Ignore this. We will not use in this class, but it is required for subclassing Optimizer.
                Defaults to None.
        """
        for group in self.param_groups:
            lr = group['lr']

            for param in group['params']:
                if param.grad is not None:
                    param.data -= lr * param.grad.data