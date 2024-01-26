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
import torch.nn as nn
from tqdm import tqdm, trange
from common.evaluators.bert_evaluator import BertEvaluator
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from common.trainers.bert_trainer import BertTrainer
from transformers.optimization import AdamW
from args import get_args
from pytorch_pretrained_bert import BertModel
import copy
import json
from transformers.tokenization_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets a list of possible labels in the dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _clean_str(cls,string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        lines = []
        with open(input_file,'r') as f:
            text = json.load(f)
            for key,value in text.items():
                lines.append(key + '\t' + value)
        return lines

class AAPDProcessor(BertProcessor):
    NAME = 'AAPD'
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir)), 'train')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            line = line.split('\t')
            label = line[0]
            text = line[1]
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):

        token = tokenizer.tokenize(example.text_a)
        if len(token) > max_seq_length - 2:
            token = token[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + token + ["[SEP]"]
        segment_ids = [0] * len(tokens)
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

        label_id = example.label

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))

    return features


class ClassifyModel(nn.Module):
    def __init__(self,pretrained_model_name_or_path,is_lock = False):
        super(ClassifyModel,self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)

        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids = None,attention_mask = None,label = None,):
        local_text,global_text = self.bert(input_ids,token_type_ids,attention_mask,output_all_encoded_layers = False)   #4*256*768

        return global_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrain_model_dir = '/home/ltf/code/data/bert-base-uncased/'
tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=True)
model = ClassifyModel(pretrain_model_dir, is_lock=True)
model.to(device)

processor =  AAPDProcessor()
train_examples = processor.get_train_examples('/home/ltf/code/data/BenchmarkingZeroShot/synsets_80.json')
train_features = convert_examples_to_features(train_examples, 100, tokenizer)

unpadded_input_ids = [f.input_ids for f in train_features]
unpadded_input_mask = [f.input_mask for f in train_features]
unpadded_segment_ids = [f.segment_ids for f in train_features]

padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)

assert len(padded_input_ids) == len(padded_input_mask) == len(padded_segment_ids)

eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=80,drop_last = True)

model.eval()

for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating", disable=False):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids.view(-1, 100), segment_ids.view(-1, 100), input_mask.view(-1, 100))