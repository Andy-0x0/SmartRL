from typing import Any, Callable
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.modules.loss as L
import torch.nn.functional as F
import torch.distributions as D

from Prototype import Agent, Environment
from Tools import ReplayBuffer, PERBuffer, Debugger, Recorder
from Trainer import optimizer


__all__ = [
    'DQNAgent',
    'DoubleDQNAgent',
    'REINFORCEAgent'
]


# DQN Agent
# TODO: This is just a discrete DQN agent. Need to be generalized to a self-defined DQN agent later
class DQNAgent(Agent):
    def __init__(
        self,
        opt_policy: nn.Module,
        smp_policy: nn.Module,
        gamma: float=0.99,
        epsilon: float=0.1,
        loss: str | Callable='MSELoss',
        lr: float=1e-3,
        optimizer: str='AdamW',
        buffer_size: int=10000,
        batch_size: int=128,
        sync_freq: int=20,
        device: str | torch.device='cpu',
        seed: int=42,
    ) -> None:
        super().__init__()

        self.opt_policy = opt_policy
        self.smp_policy = smp_policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.sync_freq = sync_freq

        self.tick = 0
        self.debugger = Debugger(level="INFO")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None or device not in ('cpu', 'cuda') else torch.device(device)

        self.set_seed(seed)
        self.action_space = None
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, batch_size=batch_size, seed=seed)

        self._config()

    def _config(self):
        if isinstance(self.device, str):
            if self.device in ('cuda', 'cpu'):
                self.device = torch.device(self.device)
            else:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not isinstance(self.device, torch.device):
            self.debugger.error(role='Agent', message=f"Invalid device {self.device}")
            raise TypeError('device is not a torch.device')

        self.opt_policy = self.opt_policy.to(device=self.device)
        self.smp_policy = self.smp_policy.to(device=self.device)
        self.optimizer = getattr(optim, self.optimizer)(self.opt_policy.parameters(), lr=self.lr)

        if isinstance(self.loss, str):
            self.loss = getattr(L, self.loss)()

        @optimizer(self.optimizer, self.debugger)
        def loss_funct(*args, **kwargs):
            return self.loss(*args, **kwargs)

        setattr(self, "optimize", loss_funct)

    def set_level(self, level:str) -> None:
        self.debugger.set_level(level)
        self.replay_buffer.set_level(level)

    def register(
        self,
        env: Environment,
        *args,
        **kwargs
    ) -> bool:
        """
        Register the environment to the agent

        :param env:     The environment to register
        :param args:    Additional positional arguments related to the environment registration
        :param kwargs:  Additional keyword arguments related to the environment registration
        :return:
        """
        if hasattr(env, 'get_action_space'):
            self.action_space = env.get_action_space()
        else:
            raise ValueError('Environment does not have an action space, please instantiate the environment using the Environment interface')

        return True

    def sync_policy(self) -> None:
        """
        Synchronize the behavior policy with the target policy

        :return: None
        """
        self.smp_policy = deepcopy(self.opt_policy)

    def tick_step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        *args,
        **kwargs
    ) -> Any:
        """
        Update the policy of the agent

        :param state:   The current state of the environment
        :param action:  The action taken by the agent
        :param reward:  The reward obtained by the agent
        :param next_state: The next state of the environment
        :param done:    Whether the episode is done or not
        :param args:    Additional positional arguments to update the policy
        :param kwargs:  Additional keyword arguments to update the policy

        :return: The loss value
        """
        if self.tick % self.sync_freq == 0:
            self.sync_policy()
        self.tick += 1

        # Get a mini-batch from the replay buffer
        self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        batch_data = self.replay_buffer.batch()

        if not self.replay_buffer.is_ready():
            return {'loss': np.nan, 'reward': reward}

        # Q(S_{t}, A_{t})
        q_predicts = self.opt_policy(torch.tensor(batch_data['state'], dtype=torch.float, device=self.device))
        q_predict = torch.gather(
            input=q_predicts,
            dim=1,
            index=torch.tensor(batch_data['action'], dtype=torch.int64, device=self.device).view(-1, 1)
        )

        # max(Q(S_{t+1}, A_{t+1}))
        q_targets = self.smp_policy(torch.tensor(batch_data['next_state'], dtype=torch.float, device=self.device)).detach()
        q_target = q_targets.max(dim=1)[0].view(-1, 1)

        td_target = torch.tensor(batch_data['reward'], dtype=torch.float, device=self.device).view(-1, 1) + self.gamma * q_target * (1 - torch.tensor(batch_data['done'], dtype=torch.float, device=self.device).view(-1, 1))
        td_predict = q_predict

        # loss(R_{t} + \gamma * max(Q(S_{t}, A_{t})), Q(S_{t+1}, A_{t+1}))
        info = {
            'loss': self.optimize(td_predict, td_target).item(),
            'reward': reward,
        }
        return info

    def episode_step(
       self,
       *args,
       **kwargs
    ) -> Any:
        """
        Placeholder method for episode step

        :param args:    No direct meaning currently
        :param kwargs:  No direct meaning currently

        :return: None
        """
        return {}

    def act(
        self,
        state,
        *args,
        **kwargs
    ) -> Any:
        """
        Take the action according to the current policy and using epsilon-greedy on the top

        :param state:   The current state of the environment
        :param args:    Additional position arguments to take the action
        :param kwargs:  Additional keyword arguments to take the action

        :return: The action taken by the agent
        """
        if np.random.random() < self.epsilon or not self.replay_buffer.is_ready():
            return np.random.choice(len(self.action_space))
        else:
            self.smp_policy.eval()
            actions = self.smp_policy(torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1)).detach()
            self.smp_policy.train()

            return torch.argmax(actions).item()


# Double DQN Agent
# TODO: This is just a discrete Double DQN agent. Need to be generalized to a self-defined Double DQN agent later
class DoubleDQNAgent(Agent):
    def __init__(
        self,
        opt_policy: nn.Module,
        smp_policy: nn.Module,
        gamma: float=0.99,
        epsilon: float=0.1,
        loss: str | Callable='MSELoss',
        lr: float=1e-3,
        optimizer: str='AdamW',
        buffer_size: int=10000,
        batch_size: int=128,
        sync_freq: int=20,
        device: str | torch.device='auto',
        seed: int=42,
    ) -> None:
        super().__init__()
        self.opt_policy = opt_policy
        self.smp_policy = smp_policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss = loss
        self.sync_freq = sync_freq

        self.tick = 0
        self.optimizer = optimizer
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None or device not in ('cpu', 'cuda') else torch.device(device)
        self.debugger = Debugger(level="INFO")

        self.set_seed(seed)
        self.action_space = None
        self.replay_buffer = PERBuffer(capacity=buffer_size, batch_size=batch_size, seed=seed)

        self._config()

    def _config(self):
        if isinstance(self.device, str):
            if self.device in ('cuda', 'cpu'):
                self.device = torch.device(self.device)
            else:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not isinstance(self.device, torch.device):
            self.debugger.error(role='Agent', message=f"Invalid device {self.device}")
            raise TypeError('device is not a torch.device')

        self.opt_policy = self.opt_policy.to(device=self.device)
        self.smp_policy = self.smp_policy.to(device=self.device)
        self.optimizer = getattr(optim, self.optimizer)(self.opt_policy.parameters(), lr=self.lr)

        if isinstance(self.loss, str):
            self.loss = getattr(L, self.loss)()

        @optimizer(self.optimizer, self.debugger)
        def loss_funct(*args, **kwargs):
            return self.loss(*args, **kwargs)

        setattr(self, "optimize", loss_funct)

    def set_level(self, level: str) ->  None:
        self.debugger.set_level(level)
        self.replay_buffer.set_level(level)

    def register(
        self,
        env: Environment,
        *args,
        **kwargs
    ) -> bool:
        """
        Register the environment to the agent

        :param env:     The environment to register
        :param args:    Additional positional arguments related to the environment registration
        :param kwargs:  Additional keyword arguments related to the environment registration
        :return:
        """
        if hasattr(env, 'get_action_space'):
            self.action_space = env.get_action_space()
        else:
            raise ValueError('Environment does not have an action space, please instantiate the environment using the Environment interface')

        return True

    def sync_policy(self) -> None:
        """
        Synchronize the behavior policy with the target policy

        :return: None
        """
        self.smp_policy = deepcopy(self.opt_policy)

    def tick_step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        *args,
        **kwargs
    ) -> Any:
        """
        Update the policy of the agent

        :param state:   The current state of the environment
        :param action:  The action taken by the agent
        :param reward:  The reward obtained by the agent
        :param next_state: The next state of the environment
        :param done:    Whether the episode is done or not
        :param args:    Additional positional arguments to update the policy
        :param kwargs:  Additional keyword arguments to update the policy

        :return: The loss value
        """
        if self.tick % self.sync_freq == 0:
            self.sync_policy()
        self.tick += 1

        # Get a mini-batch from the replay buffer
        # |R_{t} + \gamma * max(Q(S_{t}, A_{t})) - Q(S_{t+1}, A_{t+1})|
        q_predict = self.opt_policy(torch.tensor(state, dtype=torch.float, device=self.device))[action].detach().cpu()
        q_target = self.smp_policy(torch.tensor(next_state, dtype=torch.float, device=self.device)).detach().cpu()
        q_target = reward + self.gamma * q_target.max(dim=0)[0].view(-1, 1) * (1 - int(done))
        sigma = torch.abs(q_target - q_predict).item()

        self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done, sigma=sigma)
        batch_data = self.replay_buffer.batch()

        if not self.replay_buffer.is_ready():
            return {'loss': np.nan, 'reward': reward}

        # Q(S_{t}, A_{t})
        q_predicts = self.opt_policy(torch.tensor(batch_data['state'], dtype=torch.float, device=self.device))
        q_predict = torch.gather(
            input=q_predicts,
            dim=1,
            index=torch.tensor(batch_data['action'], dtype=torch.int64, device=self.device).view(-1, 1)
        )

        # Q(S_{t+1}, argmax(Q(S_{t+1}, A_{t+1})))
        q_targets = self.smp_policy(torch.tensor(batch_data['next_state'], dtype=torch.float, device=self.device)).detach()
        q_target = torch.gather(
            input=q_targets,
            dim=1,
            index=torch.argmax(
                self.opt_policy(torch.tensor(batch_data['next_state'], dtype=torch.float, device=self.device)),
                dim=1,
                keepdim=True
            )
        )

        td_target = torch.tensor(batch_data['reward'], dtype=torch.float, device=self.device).view(-1, 1) + self.gamma * q_target * (1 - torch.tensor(batch_data['done'], dtype=torch.float, device=self.device).view(-1, 1))
        td_predict = q_predict

        # loss(R_{t} + \gamma * max(Q(S_{t}, A_{t})), Q(S_{t+1}, A_{t+1}))
        info = {'loss': self.optimize(td_predict, td_target).item(), 'reward': reward}
        return info

    def episode_step(
       self,
       *args,
       **kwargs
    ) -> Any:
        """
        Placeholder method for episode step

        :param args:    No direct meaning currently
        :param kwargs:  No direct meaning currently

        :return: None
        """
        return {}

    def act(
        self,
        state,
        *args,
        **kwargs
    ) -> Any:
        """
        Take the action according to the current policy and using epsilon-greedy on the top

        :param state:   The current state of the environment
        :param args:    Additional position arguments to take the action
        :param kwargs:  Additional keyword arguments to take the action

        :return: The action taken by the agent
        """
        if np.random.random() < self.epsilon or not self.replay_buffer.is_ready():
            return np.random.choice(len(self.action_space))
        else:
            self.smp_policy.eval()
            actions = self.smp_policy(torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1)).detach()
            self.smp_policy.train()

            return torch.argmax(actions).item()


# REINFORCE Agent
# TODO: This is just a discrete REINFORCE agent. Need to be generalized to a self-defined REINFORCE agent later
class REINFORCEAgent(Agent):
    def __init__(
        self,
        opt_policy: nn.Module,
        val_policy: nn.Module,
        gamma: float=0.99,
        lr_p: float=1e-3,
        lr_v: float=1e-2,
        optimizer_p: str='AdamW',
        optimizer_v: str='AdamW',
        device: str | torch.device='auto',
        seed: int=42,
    ) -> None:
        super().__init__()
        self.opt_policy = opt_policy
        self.val_policy = val_policy
        self.gamma = gamma
        self.device = device
        self.optimizer_p = optimizer_p
        self.optimizer_v = optimizer_v
        self.lr_p = lr_p
        self.lr_v = lr_v

        self.recorder_tick = Recorder()
        self.recorder_prob = Recorder()
        self.debugger = Debugger(level="INFO")

        self.set_seed(seed)
        self.action_space = None

        self._config()

    def _config(self):
        if isinstance(self.device, str):
            if self.device in ('cuda', 'cpu'):
                self.device = torch.device(self.device)
            else:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not isinstance(self.device, torch.device):
            self.debugger.error(role='Agent', message=f"Invalid device {self.device}")
            raise TypeError('device is not a torch.device')

        self.debugger.info(role='Agent', message=f"Training on device {self.device}")

        self.opt_policy = self.opt_policy.to(self.device)
        self.val_policy = self.val_policy.to(self.device)
        self.optimizer_p = getattr(optim, self.optimizer_p)(self.opt_policy.parameters(), lr=self.lr_p)
        self.optimizer_v = getattr(optim, self.optimizer_v)(self.val_policy.parameters(), lr=self.lr_v)

        @optimizer(self.optimizer_v, self.debugger)
        def loss_funct_v(*args, **kwargs):
            return L.MSELoss()(*args, **kwargs)

        @optimizer(self.optimizer_p, self.debugger)
        def loss_funct_p(Gs, values, probs):
            loss = 0
            for G, val, prob in zip(Gs, values, probs):
                loss -= torch.log(prob) * (G - val)

            return loss

        setattr(self, "optimize_v", loss_funct_v)
        setattr(self, "optimize_p", loss_funct_p)

    def register(
        self,
        env: Environment,
        *args,
        **kwargs
    ) -> bool:
        """
        Register the environment to the agent

        :param env:     The environment to register
        :param args:    Additional positional arguments related to the environment registration
        :param kwargs:  Additional keyword arguments related to the environment registration
        :return:
        """
        if hasattr(env, 'get_action_space'):
            self.action_space = env.get_action_space()
        else:
            raise ValueError('Environment does not have an action space, please instantiate the environment using the Environment interface')

        return True

    def tick_step(
        self,
        state,
        reward,
        done,
        *args,
        **kwargs
    ) -> Any:
        """
        Update the policy of the agent

        :param args:    Additional positional arguments to update the policy
        :param kwargs:  Additional keyword arguments to update the policy

        :return: The loss value
        """
        self.recorder_tick.append(
            state=state,
            reward=reward,
            done=bool(done)
        )

        return {'reward': reward}


    def episode_step(
        self,
        *args,
        **kwargs
    ) -> Any:
        """
        Update the policy of the agent

        :param args:    Additional positional arguments to update the policy
        :param kwargs:  Additional keyword arguments to update the policy

        :return: The loss value
        """
        G = 0
        Gs = []
        values = []

        for srd, prob in zip(reversed(self.recorder_tick), reversed(self.recorder_prob)):
            G = srd['reward'] + self.gamma * G * (1 - int(srd['done']))
            Gs.append(G)
            values.append(self.val_policy(torch.tensor(srd['state'], dtype=torch.float, device=self.device)).squeeze())
        Gs.reverse()
        values.reverse()

        info = {
            'value loss': self.optimize_v(
                torch.stack(values, dim=0),
                torch.tensor(Gs, dtype=torch.float, device=self.device)
            ).item(),

            'policy loss':  self.optimize_p(
                torch.tensor(Gs, dtype=torch.float, device=self.device), 
                [v.detach() for v in values],
                self.recorder_prob['prob'],
            ).item()
        }

        self.recorder_tick.clear()
        self.recorder_prob.clear()

        return info

    def act(
        self,
        state,
        *args,
        **kwargs
    ) -> Any:
        """
        Take the action according to the current policy and using epsilon-greedy on the top

        :param state:   The current state of the environment
        :param args:    Additional position arguments to take the action
        :param kwargs:  Additional keyword arguments to take the action

        :return: The action taken by the agent
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1)
        action_probs = self.opt_policy(state).squeeze()
        action = D.Categorical(action_probs).sample().item()
        prob = action_probs[action]

        self.recorder_prob.append(prob=prob)

        return action

    def set_level(self, level: str) ->  None:
        self.debugger.set_level(level)
        self.recorder_tick.set_level(level)
        self.recorder_prob.set_level(level)

