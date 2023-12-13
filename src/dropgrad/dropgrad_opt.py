from typing import Callable, Dict, Optional
import torch
from torch.optim import Optimizer

class DropGrad(object):
    """DropGrad is a wrapper around an optimizer that drops gradients according to a specified probability.

    Args:
        optimizer (Optimizer): The optimizer to wrap.
        drop_rate (float, optional): The probability of dropping a gradient. Defaults to None.
        params (Dict[torch.nn.Parameter, float], optional): A dictionary mapping parameters to drop rates. Defaults to None.
    """
    def __init__(self, 
                 optimizer: Optimizer, 
                 drop_rate: Optional[float] = None, 
                 params: Optional[Dict[torch.nn.Parameter, float]] = None
        ):
        if drop_rate is None and params is None:
            raise ValueError("Either drop_rate or params must be specified")
        if params is not None:
            for value in params.values():
                if value < 0.0 or value > 1.0:
                    raise ValueError("Drop rate must be in [0.0, 1.0]")
        if drop_rate is not None:
            if drop_rate < 0.0 or drop_rate > 1.0:
                raise ValueError("Drop rate must be in [0.0, 1.0]")
            
        self.optimizer = optimizer
        self.drop_rate = drop_rate
        self.params = params

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (Optional[Callable[[], float]], optional): A closure that reevaluates the model and returns the loss. Defaults to None.

        Returns:
            Optional[float]: The loss returned by the closure, if any.
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if self.params is not None:
                        drop_rate = self.params.get(param, self.drop_rate)
                    else:
                        drop_rate = self.drop_rate
                        
                    if drop_rate is not None:
                        param.grad.data.mul_(
                            torch.bernoulli(
                                torch.full_like(param.grad.data, 1.0 - drop_rate)
                            )
                        ).div_(1.0 - drop_rate)
                    
        return self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clears the gradients of all optimized parameters.
        
        Args:
            set_to_none (bool, optional): Whether to set the gradients to None. Defaults to True.
        Returns:
            None
        """
        
        return self.optimizer.zero_grad(set_to_none)
        