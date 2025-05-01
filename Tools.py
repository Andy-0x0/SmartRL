import os
from typing import Any, Dict, List, Sequence, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import time
import gc

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon


# Debug configuration
class Debugger:
    def __init__(self, level="INFO") -> None:
        self.level_map = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
        }
        self.level = self.level_map[level]
        self.last_time_stamp = None

    @property
    def _time_stamp(self) -> str:
        curr_time_stamp = time.strftime("[%m/%d/%y %H:%M:%S]", time.localtime())
        if self.last_time_stamp is None or self.last_time_stamp != curr_time_stamp:
            self.last_time_stamp = curr_time_stamp
            return curr_time_stamp
        else:
            return ' ' * len(curr_time_stamp)

    @staticmethod
    def _role(role: str) -> str:
        return f"[{role}]"

    @staticmethod
    def _level(level: str) -> str:
        return f"{level:9s}"

    def _do_print(self, level_digit: int) -> bool:
        return level_digit >= self.level

    @staticmethod
    def _red(text: str) -> str:
        return f"\033[91m{str(text)}\033[0m"

    @staticmethod
    def _green(text:str) -> str:
        return f"\033[92m{str(text)}\033[0m"

    @staticmethod
    def _yellow(text:str) -> str:
        return f"\033[93m{str(text)}\033[0m"

    @staticmethod
    def _blue(text:str) -> str:
        return f"\033[94m{str(text)}\033[0m"

    def set_level(self, level: str) -> None:
        self.level = self.level_map[level]

    def info(self, role: str, message: str) -> str:
        if self._do_print(1):
            text = f"{self._time_stamp} {self._blue(self._level('INFO'))}{Debugger._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''

    def debug(self, role: str, message: str) -> str:
        if self._do_print(0):
            text = f"{self._time_stamp} {self._green(self._level('DEBUG'))}{Debugger._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''

    def warning(self, role: str, message: str) -> str:
        if self._do_print(2):
            text = f"{self._time_stamp} {self._yellow(self._level('WARNING'))}{Debugger._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''

    def error(self, role: str, message: str) -> str:
        if self._do_print(3):
            text = f"{self._time_stamp} {self._red(self._level('ERROR'))}{Debugger._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''


@dataclass(repr=True)
class CheckPoint:
    model_skeletons: Dict[str, nn.Module] = field(repr=False)
    model_parameters: Dict[str, nn.Parameter] = field(repr=False)
    name: str = field(default="checkpoint", repr=True)
    folder: str = field(
        default=f"{os.getcwd()}/checkpoints",
        repr=True
    )


class CheckpointBot:
    def __init__(
        self,
        checkpoint_path: str,
        debugger: Debugger
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.debugger = debugger

    def load(
        self,
        path: str | os.PathLike
    ) -> CheckPoint:
        pass

    def pack(
        self,
        *args,
        **kwargs
    ) -> CheckPoint:
        pass

    def save(
        self,
        checkpoint: CheckPoint,
        path: str | os.PathLike
    ) -> None:
        pass


# Replay Buffer for DQN
class ReplayBuffer:
    def __init__(
        self,
        capacity: int=10000,
        batch_size: int=128,
        seed: int=42
    ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.seed = seed
        self.keys = None
        self.buffer = deque(maxlen=capacity)
        self.debugger = Debugger(level="DEBUG")

        self.set_seed(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f'ReplayBuffer(capacity={self.capacity}, batch_size={self.batch_size}, seed={self.seed})'

    @staticmethod
    def _tolist(
        obj: Any | List | np.ndarray | torch.Tensor
    ) -> Any:
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        else:
            return obj

    def is_ready(self) -> bool:
        return len(self) >= self.batch_size

    def set_level(self, level: str) -> None:
        self.debugger.set_level(level)

    def set_seed(
        self,
        seed: int=42
    ) -> None:
        """
        Set the random seed for the replay buffer

        :param seed: The random seed

        :return: None
        """
        self.seed = seed
        # random.seed(self.seed)
        # np.random.seed(self.seed)

        self.debugger.debug(role="ReplayBuffer", message=f'Setting seed to {seed}')

    def add(
        self,
        **kwargs
    ) -> None:
        """
        Add a new sample to the replay buffer

        :param kwargs: The key-value pairs of the sample to add

        :return: None
        """
        if self.keys is None:
            self.keys = list(kwargs.keys())
            self.debugger.debug(role="ReplayBuffer", message=f'Collecting {sorted(self.keys)}')
        else:
            if self.keys != list(kwargs.keys()):
                error_keys = sorted(list((set(kwargs.keys()) | set(self.keys)) - (set(kwargs.keys()) & set(self.keys))))
                self.debugger.error(role="ReplayBuffer", message=f'Missing/Unexpected keys {error_keys}')
                raise ValueError(f'Missing/unexpected keys to the replay buffer {error_keys}')

        self.buffer.append({
            key: ReplayBuffer._tolist(value) for key, value in kwargs.items()
        })

    def batch(
        self
    ) -> Dict:
        """
        Get a mini-batch of samples from the replay buffer

        :return: the mini-batch of samples
        """
        if len(self.buffer) < self.batch_size:
            batch_data = self.buffer
            size = len(self.buffer)
        else:
            batch_data = random.sample(self.buffer, self.batch_size)
            size = self.batch_size

        batch_data = {key: [batch_data[i][key] for i in range(size)] for key in self.keys}

        return batch_data


# Priority Replay Buffer for DoubleDQN
class PERBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 128,
        seed: int = 42
    ):
        super().__init__(capacity=capacity, batch_size=batch_size, seed=seed)
        self.capacity = capacity
        self.batch_size = batch_size
        self.seed = seed
        self.keys = None
        self.buffer = deque(maxlen=capacity)
        self.updated = False
        self.probabilities = np.array([], dtype=np.float32)
        self.debugger = Debugger(level="DEBUG")

    def add(
        self,
        sigma: float,
        **kwargs
    ) -> None:
        kwargs['sigma'] = sigma

        if self.keys is None:
            self.keys = list(kwargs.keys())
            self.debugger.debug(role="ReplayBuffer", message=f'Collecting {sorted(self.keys)}')
        else:
            if self.keys != list(kwargs.keys()):
                error_keys = sorted(list((set(kwargs.keys()) | set(self.keys)) - (set(kwargs.keys()) & set(self.keys))))
                self.debugger.error(role="ReplayBuffer", message=f'Missing/Unexpected keys {error_keys}')
                raise ValueError(f'Missing/unexpected keys to the replay buffer {error_keys}')

        self.buffer.append({
            key: ReplayBuffer._tolist(value) for key, value in kwargs.items()
        })

        self.updated = True

    def batch(
        self
    ) -> Dict:
        if len(self.buffer) < self.batch_size:
            batch_data = self.buffer
            size = len(self.buffer)
        else:
            if self.updated:
                logits = np.array([self.buffer[i]['sigma'] for i in range(len(self))])
                self.probabilities = logits / np.sum(logits, axis=0)
                self.updated = False

            batch_data = np.random.choice(self.buffer, self.batch_size, p=self.probabilities)
            size = self.batch_size

        batch_data = {key: [batch_data[i][key] for i in range(size)] for key in self.keys}

        return batch_data


class Recorder:
    def __init__(
        self,
        keys=None,
        level='ERROR'
    ) -> None:
        self.memory = []
        self._keys = keys
        self.debugger = Debugger(level=level)

    def __repr__(self) -> str:
        return f"Recorder(keys={self._keys}, __len__={len(self)})"

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, idx: Any) -> Dict | List:
        if isinstance(idx, int):
            return self.memory[idx]
        else:
            return [self.memory[i][idx] for i in range(len(self))]

    def keys(self) -> Tuple:
        return tuple(self._keys)

    def values(self) -> Tuple:
        vals = [[self.memory[i][key] for i in range(len(self))] for key in self._keys]
        return tuple(vals)

    def items(self) -> Tuple:
        pairs = [(key, [self.memory[i][key] for i in range(len(self))]) for key in self._keys]
        return tuple(pairs)

    def append(
        self,
        **kwargs
    ) -> None:
        """
        Add a new sample to the replay buffer

        :param kwargs: The key-value pairs of the sample to add

        :return: None
        """
        if self._keys is None:
            self._keys = list(kwargs.keys())
            self.debugger.debug(role="Recorder", message=f'Collecting {sorted(self._keys)}')
            self.memory.append({
                key: value for key, value in kwargs.items()
            })
        else:
            if self._keys != list(kwargs.keys()):
                self.debugger.warning(role="Recorder", message=f'Missing/Unexpected keys appended')
                temp_dict = {key: None for key in self._keys}
                for key, value in kwargs.items():
                    temp_dict[key] = value

                self.memory.append(temp_dict)
                del temp_dict
            else:
                self.memory.append({
                    key: value for key, value in kwargs.items()
                })

    def clear(self) -> None:
        self.debugger.debug(role="Recorder", message=f'Emptying {len(self)} samples')
        self.memory = []
        self._keys = None
        gc.collect()

    def set_level(self, level: str) -> None:
        self.debugger.set_level(level=level)


import os
from typing import Any, Dict, List, Sequence, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import random
import time
import gc

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon


def fill_alpha(
    x: Sequence[float | int],
    y_top: Sequence[float | int],
    y_low: Optional[float | int | str] = 'auto_down',
    ratio: Optional[float] = 0.7,
    alpha: Optional[float] = 0.5,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    def to_numpy(obj):
        if not isinstance(obj, Sequence):
            return np.ones(len(x)) * obj
        elif isinstance(obj, torch.Tensor):
            return obj.numpy()
        elif isinstance(obj, pd.Series):
            return obj.to_numpy()
        else:
            return np.array(obj)

    # meta information
    x = to_numpy(x)
    y_top = to_numpy(y_top)
    height = 1000
    width = len(x)

    y_top_max, y_top_min = np.nanmax(y_top), np.nanmin(y_top)
    local_extreme = abs(y_top_max - y_top_min)

    if isinstance(y_low, str) and y_low == 'auto_down':
        y_low = y_top_min - ratio * local_extreme
    elif isinstance(y_low, str) and y_low == 'auto_up':
        y_low = y_top_max + ratio * local_extreme

    y_abs_max, y_abs_min = max(y_low, y_top_max), min(y_low, y_top_min)
    global_extreme = abs(y_abs_max - y_abs_min)

    x_max, x_min = np.nanmax(x), np.nanmin(x)

    if ax is None:
        ax = plt.gca()
        ax.set_ylim(y_abs_min - global_extreme * 0.3, y_abs_max + global_extreme * 0.3)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))

    line, = ax.plot(x, y_top, color=color, **kwargs)
    if color is None:
        color = line.get_color()

    zorder = line.get_zorder()
    alpha = 0.5 if alpha is None else alpha

    unit = height / global_extreme
    if y_top_min <= y_low <= y_top_max:
        gradient_size = int(ratio * local_extreme * unit)
    else:
        gradient_size = int(ratio * abs(y_low - y_top_min) * unit)

    # calculate and map the alpha onto the image background
    def map_alpha_down(index):
        occ_size = int((y_top_max - y_top[index]) * unit)
        emp_size = height - occ_size - gradient_size
        ans = np.hstack([
            np.zeros(abs(emp_size)),
            np.linspace(0, alpha, gradient_size),
            np.ones(abs(occ_size)) * alpha
        ])

        return ans[-height:]

    def map_alpha_up(index):
        occ_size = int((y_top[index] - y_top_min) * unit)
        emp_size = height - occ_size - gradient_size
        ans = np.hstack([
            np.ones(abs(occ_size)) * alpha,
            np.linspace(alpha, 0, gradient_size),
            np.zeros(abs(emp_size))
        ])

        return ans[: height]

    def map_alpha(index):
        if y_top[index] >= y_low:
            return map_alpha_down(index)
        else:
            return map_alpha_up(index)

    # compute the pixels for the color and alpha mapping
    pixel = np.zeros((height, width, 4), dtype=np.float32)
    pixel[:, :, :3] = mcolors.colorConverter.to_rgb(color)
    for i in range(width):
        pixel[:, i, 3] = map_alpha(i)
    pixel[:, :, 3] = gaussian_filter(pixel[:, :, 3], sigma=(10, 10))

    # clockwise path planning
    curve = np.column_stack([x, y_top])
    outline = np.vstack([
        [x_min, y_low],
        curve,
        [x_max, y_low],
        [x_min, y_low]
    ])
    clip_path = Polygon(outline, facecolor='none', edgecolor='none', closed=True)

    # mapping and clipping
    im = ax.imshow(
        pixel,
        aspect='auto',
        extent=(np.nanmin(x), np.nanmax(x), min(y_low, y_top_min), max(y_low, y_top_max)),
        origin='lower',
        zorder=zorder,
    )
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    ax.autoscale(True)

    return ax