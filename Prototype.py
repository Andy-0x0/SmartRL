from abc import ABC, abstractmethod
from Tools import Debugger
from typing import Any, Tuple
import numpy as np
import torch

# Base class for all environments to inherit from
class Environment(ABC):
    @abstractmethod
    def step(
        self,
        action,
        *args,
        **kwargs
    ) -> Tuple[Any, Any, bool]:
        """
        Update the inner state of the environment

        :param action:  ...
        :return: None
        """
        pass

    @abstractmethod
    def reset(
        self,
        *args,
        **kwargs
    ) -> Any:
        """
        Reset the environment state to the initial state

        :return: The initial state
        """
        pass

    @abstractmethod
    def get_action_space(
        self,
        *args,
        **kwargs
    ) -> Any:
        """
        Get the action space of the environment

        :param args:
        :param kwargs:
        :return:
        """
        pass


# Base class for all agents to inherit from
class Agent(ABC):
    def __init__(self):
        self.seed = None
        self.debugger = Debugger(level='INFO')

    def set_seed(
        self,
        seed
    ) -> None:
        """
        Set the local random seed for the agent

        :param seed: The random seed
        :return: None
        """
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def set_level(self, level: str) ->  None:
        """
        Set the level of the logging information

        :param level: The logging level

        :return: None
        """
        self.debugger.set_level(level)

    @abstractmethod
    def tick_step(
        self,
        *args,
        **kwargs
    ) -> Any:
        """
        Inner update on change of each action cycle

        :param args:    The position arguments to update the policy
        :param kwargs:  The keyword arguments to update the policy
        :return: None
        """
        pass

    @abstractmethod
    def episode_step(
       self,
       *args,
       **kwargs
    ) -> Any:
        """
        Inner update on change of each episode cycle

        :param args:    The position arguments to update the policy
        :param kwargs:  The keyword arguments to update the policy
        :return: None
        """
        pass

    @abstractmethod
    def act(
       self,
       *args,
       **kwargs
    ) -> Any:
        """
        Get the action according to the current policy

        :param args:    The position arguments to take the action
        :param kwargs:  The keyword arguments to take the action

        :return:
        """
        pass

    @abstractmethod
    def register(
        self,
        env: Environment,
        *args,
        **kwargs
    ) -> Any:
        """
        Register the environment to the agent

        :param env:     The environment to register
        :param args:    The position arguments related to the environment registration
        :param kwargs:  The keyword arguments related to the environment registration

        :return:        Any information related to the environment registration, None if there's no need
        """
        pass







