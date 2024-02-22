import io
import logging
import os
import posixpath
import sys
import tempfile
import time
import warnings
from logging import debug, info
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import math

import numpy as np
import yaml

def read_yamls(dir):
    conf = {}
    no_conf = True
    for config_file in Path(dir).glob('*.yaml'):
        no_conf = False
        with config_file.open('r') as f:
            conf.update(yaml.safe_load(f))
    if no_conf:
        print(f'WARNING: No yaml files found in {dir}')
    return conf

def to_uint8(images: np.ndarray) -> np.ndarray:
    return np.clip(255*(images + 1)*0.5, 0, 255).astype(np.uint8)
