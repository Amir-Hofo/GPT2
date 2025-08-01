from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers import processors, decoders

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.data import Dataset, DataLoader, IterableDataset

import numpy as np

import os
import math
from itertools import chain