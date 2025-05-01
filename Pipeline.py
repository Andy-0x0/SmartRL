import wandb
import os
from typing import Dict, Callable, Any
from functools import partialmethod
from Tools import Debugger, Recorder, fill_alpha
from Prototype import Environment, Agent
import numpy as np
import pandas as pd
from rich.progress import Progress
import matplotlib.pyplot as plt
import gc

class Pipeline:
    def __init__(
        self,
        env: Environment,
        agent: Agent,
        episodic: bool=True,
        episode_num: int=500,
        episode_limit: int=1000,
    ) -> None:
        self.env = env
        self.agent = agent
        self.episodic = episodic
        self.episode_num = episode_num
        self.episode_limit = episode_limit
        self.debugger = None

        self._config()

    def _config(self) -> None:
        self.agent.register(env=self.env)

    def _setup_backend(
        self,
        backend:str = 'local'
    ) -> None:
        self.debugger.info(role="Pipeline", message=f"Set up backend at <{backend}>")
        if backend is not None:
            setattr(self, 'backend_recorder', Recorder(level="ERROR"))
        if backend == 'local' or backend == 'all':
            setattr(self, 'use_local', True)
        if backend == 'wandb' or backend == 'all':
            setattr(self, 'use_wandb', True)
            setattr(self, 'step_counter', 1)
            wandb.login()
            project_name = input("Please enter the project name: ")
            experiment_name = input("Please enter the experiment name: ")
            wandb.init(project=str(project_name), name=str(experiment_name))

    def _backend_step(
        self,
        **kwargs
    ) -> None:
        self.debugger.debug(role="Pipeline", message=f"backend updated")
        if hasattr(self, 'backend_recorder'):
            backend_recorder = getattr(self, 'backend_recorder')
            backend_recorder.append(**kwargs)

        if getattr(self, 'use_wandb', False):
            wandb.log(kwargs, step=getattr(self, 'step_counter'))
            self.step_counter += 1

    def _complete_backend(
        self
    ) -> Any:
        if getattr(self, 'use_local', False):
            def find_targets(array, funct):
                # Preparing window object
                left, right = 0, 0
                windows = []

                # Expanding the window
                while right < len(array):
                    ele_right = array[right]
                    right += 1

                    while left < right and (not funct(ele_right) or right == len(array)):
                        if funct(array[left]):
                            if right == len(array) and funct(ele_right):
                                windows.append((left, right))
                            else:
                                windows.append((left, right - 1))

                        left = right

                return windows

            backend_recorder = getattr(self, 'backend_recorder')
            factor_num = len(backend_recorder.keys())
            cols = int(min(3, factor_num))
            rows = int(np.ceil(factor_num / cols))
            plt.figure(figsize=(12 + cols * 4, 5 + rows * 4))
            plt.subplots_adjust(
                left=0.05,
                right=0.95,
                wspace=0.2,
                hspace=0.5
            )

            counter = 1
            for key, value in backend_recorder.items():
                ax = plt.subplot(rows, cols, counter)

                nan_intervals = find_targets(array=value, funct=np.isnan)
                val_intervals = find_targets(array=value, funct=lambda x: not np.isnan(x))
                value = pd.Series(value).bfill().ffill().to_list()
                extreme = max(value) - min(value)
                ax.set_ylim(min(value) - 0.3 * extreme, max(value) + 0.3 * extreme)

                for start, end in val_intervals:
                    ax = fill_alpha(np.arange(1, end - start + 1), value[start: end], y_low=0, color='crimson')

                for start, end in nan_intervals:
                    ax = fill_alpha(np.arange(1, end - start + 1), value[start: end], y_low=0, color='crimson', linestyle='--')

                ax.set_title(str(key).title())
                ax.set_xlabel("Episodes")
                ax.set_ylabel("Value")

                counter += 1

            plt.suptitle("Experiment Panel")
            plt.show()
            plt.close()

            delattr(self, 'use_local')

        if getattr(self, 'use_wandb', False):
            wandb.finish()
            delattr(self, 'use_wandb')

        gc.collect()

        if hasattr(self, 'backend_recorder'):
            return getattr(self, 'backend_recorder')
        else:
            return None


    def run(
        self,
        tick_factors:Dict[str, Callable],
        episode_factors:Dict[str, Callable],
        level:str="INFO",
        backend: str = "local",
        save_strategy: str = "best",
        save_path: str | os.PathLike = os.getcwd(),
    ) -> None:
        self.debugger = Debugger(level=level)
        self.agent.set_level(level=level)
        self._setup_backend(backend=backend)
        tick_factor_recorder = Recorder(keys=list(tick_factors.keys()))

        with Progress(transient=True) as pbar:
            global_task = pbar.add_task("Training...", total=self.episode_num)
            local_task = pbar.add_task("Episode...", total=self.episode_limit)

            for episode in range(self.episode_num):
                state = self.env.reset()
                done = False
                tick_counter = 0

                while not done and tick_counter < self.episode_limit:
                    # Agent interact with Environment
                    action = self.agent.act(state)
                    reward, next_state, done = self.env.step(action)

                    # Tick level agent update
                    tick_info = self.agent.tick_step(state=state, reward=reward, action=action, next_state=next_state, done=done)
                    if tick_info is not None:
                        temp_dict = {key: value for key, value in tick_info.items()}
                        tick_factor_recorder.append(**temp_dict)

                    # State Transition
                    state = next_state
                    tick_counter += 1

                    # Updating the logging information
                    pbar.update(local_task, advance=1)

                # Episodic level agent update
                episode_info = self.agent.episode_step()

                for key, value in episode_info.items():
                    episode_info[key] = episode_factors[key](value)

                for key, value in tick_factor_recorder.items():
                    episode_info[key] = tick_factors[key](value)
                tick_factor_recorder.clear()

                self._backend_step(**episode_info)

                # Updating the logging information
                self.debugger.info(
                    role="Pipeline",
                    message=f"Episode {episode + 1:{len(str(self.episode_num))}} | Reward: {episode_info['reward']:.3f} | Loss: {episode_info['loss']:.3f}"
                )
                pbar.update(global_task, advance=1)
                pbar.update(local_task, completed=0)


            self._complete_backend()
            self.debugger.info(role="Pipeline", message="End of Train")


