# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The base class for Encoder
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch

from verl import DataProto

__all__ = ["BasePPOEncoder"]


class BasePPOEncoder(ABC):
    def __init__(self, config):
        """The base class for Encoder

        Args:
            config (DictConfig): a config passed to the Encoder. We expect the type to be
                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def extract_features(self, data: DataProto) -> torch.Tensor:
        """extract mutimodal features given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```?```


        """
        pass


