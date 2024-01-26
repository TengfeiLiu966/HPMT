# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import scipy.linalg

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from common.evaluators.bert_evaluator import BertEvaluator
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from common.trainers.bert_trainer import BertTrainer
from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers import WarmupLinearSchedule
from transformers.modeling_bert import BertForSequenceClassification, BertConfig#, WEIGHTS_NAME, CONFIG_NAME
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW
from topics import topic_num_map
from args import get_args
from pytorch_pretrained_bert import BertModel
from datasets.bert_processors.aapd_processor import AAPDProcessor,MultiHeadedAttention,PositionwiseFeedForward,LayerNorm,SublayerConnection,EncoderLayer
from common.constants import *
from preprocess_situation import evaluate_situation_zeroshot_TwpPhasePred
# import torch.optim as optimizer_wenpeng
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features

class AAPDProcessor(BertProcessor):
    NAME = 'AAPD'
    NUM_CLASSES = 54
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir,'Arxiv', 'Arxiv_tu_train1.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'Arxiv', 'Arxiv_tu_dev1.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'Arxiv', 'Arxiv_tu_test1.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            number = 0
            text = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

class ClassifyModel(customizedModule):
    def __init__(self,pretrained_model_name_or_path,num_labels,Encoder1,Encoder2,is_lock = False):
        super(ClassifyModel,self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, input_ids, image, token_type_ids = None,attention_mask = None,label = None,):
        local_text,global_text = self.bert(input_ids,token_type_ids,attention_mask,output_all_encoded_layers = False)   #4*256*768

        return logits

def main():
    #Set default configuration in args.py
    args = get_args()
    processor = AAPDProcessor()

    train_examples = processor.get_train_examples(args.data_dir)

    # pretrain_model_dir = '/home/ltf/code/data/bert-base-uncased/'
    pretrain_model_dir ='/home/ltf/code/data/scibert_scivocab_uncased/'
    model = ClassifyModel(pretrain_model_dir, is_lock=False)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)