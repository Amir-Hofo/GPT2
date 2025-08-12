from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers import processors, decoders

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.checkpoint import checkpoint

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import math
import json
from itertools import chain
from types import SimpleNamespace
import contextlib
from itertools import cycle
from tqdm import tqdm
from torchmetrics import MeanMetric
from rich.table import Table
from rich.console import Console
from termcolor import colored