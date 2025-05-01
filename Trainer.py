import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.modules.loss as L
from typing import Any, Dict, List, Callable

from Tools import Debugger


def optimizer(
    opt: optim,
    logger: Debugger
) -> Callable:
    def decorator(loss_func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:

            if isinstance(opt, optim.Optimizer):
                opt.zero_grad()
            else:
                logger.error(role="LossFunct", message="Optimizer is not initialized")
                raise ValueError("Optimizer is not initialized")

            loss = loss_func(*args, **kwargs)

            loss.backward()
            opt.step()

            return loss

        return wrapper
    return decorator













