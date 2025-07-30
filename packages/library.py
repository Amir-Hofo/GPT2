from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers import processors, decoders

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

import numpy as np

import os
from itertools import chain