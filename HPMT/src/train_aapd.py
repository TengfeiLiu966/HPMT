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
sys.path.append('/home/ltf/code/HPMT/datasets')
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
from bert_processors.aapd_processor import AAPDProcessor,MultiHeadedAttention,PositionwiseFeedForward,LayerNorm,SublayerConnection,EncoderLayer,SectionOne,SentenceFour
from common.constants import *
from preprocess_situation import evaluate_situation_zeroshot_TwpPhasePred
import copy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_split(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask,mask1,mask2):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x,att = layer(x, mask,mask1,mask2)
        # x = self.layers[index](x, mask)
        return self.norm(x),att

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class positionembeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self,device):
        super(positionembeddings, self).__init__()
        self.position_embeddings = nn.Embedding(10, 768)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.device = device

    def forward(self, words_embeddings):
        position_ids = torch.arange(10, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(torch.randn(words_embeddings.shape[0],10))
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl

class sentence_genaration(customizedModule):
    def __init__(self,):
        super(sentence_genaration, self).__init__()
        self.g_W2 = self.customizedLinear(768, 768)

    def forward(self, word_feature,sentence_mask,device):
        sentence_feature = []
        for i in range(4):
            nn = 0
            sentence_feature_s = []
            index = torch.argmax(sentence_mask[i, :, :].view(-1), dim=0)
            bb = sentence_mask[i, :, :].view(-1)[index].type(torch.int32)
            for j in range(bb):
                mid_mask = (sentence_mask.view(4, -1)[i, :] == j + 1).unsqueeze(-1).repeat(1, 768)
                sentence_feature_s.append(torch.max(word_feature[i, :, :][mid_mask].view(-1, 768), dim=0)[0].unsqueeze(0))
            if len(sentence_feature_s) < 100:
                sentence_feature_s.append(torch.tensor((100 - len(sentence_feature_s)) * [[0] * 768]).to(device))
            else:
                sentence_feature_s = sentence_feature_s[:100]

            sentence_feature.append(torch.cat(sentence_feature_s, dim=0).unsqueeze(0))
        sentence_feature = self.g_W2(torch.cat(sentence_feature, dim=0))
        return sentence_feature

class ClassifyModel(customizedModule):
    def __init__(self,pretrained_model_name_or_path,num_labels,args,Encoder2,device,is_lock = False):
        super(ClassifyModel,self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.args = args
        self.device = device
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,num_labels)
        self.sentence_genaration = sentence_genaration()
        self.init_mBloSA()
        self.SectionOne = SectionOne(device)
        self.SentenceFour = SentenceFour(device)
        self.Encoder2 = Encoder2
        self.missing_img_prompt = nn.Parameter(torch.zeros(4, 4, 768))

        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(512, 768)
        self.g_W3 = self.customizedLinear(768, 768)
        self.g_W4 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False

        self.f_W1 = self.customizedLinear(768, 300, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(300, 768)
        self.f_W3 = self.customizedLinear(768, 300, activation=nn.ReLU())
        self.f_W4 = self.customizedLinear(300, 768)
        self.f_W5 = self.customizedLinear(768 * 2, 768, activation=nn.ReLU())
        self.f_W6 = self.customizedLinear(768 * 2, 768)
        self.f_W7 = self.customizedLinear(768 * 2, 768, activation=nn.ReLU())
        self.f_W8 = self.customizedLinear(768 * 2, 768)
        self.f_W9 = self.customizedLinear(768 * 2, 1, activation=nn.ReLU())
        self.f_W10 = self.customizedLinear(768 * 2, 1)

    def forward(self, input_ids,image1,sentence_mask,token_type_ids = None,attention_mask = None,label = None,):
        local_text,global_text = self.bert(input_ids,token_type_ids,attention_mask,output_all_encoded_layers = False)   #4*256*768
        #文本特征********************************************************************************************************************
        section_prompt = self.f_W2(self.f_W1(self.missing_img_prompt))
        sentence_prompt = self.f_W4(self.f_W3(self.missing_img_prompt))

        #单词特征
        local_text1 = local_text[:, 1:-1, :]  # 获取段中所有单词的特征
        word_feature = torch.reshape(local_text1, (-1,4*(self.args.max_seq_length-2), 768))
        # # 句子特征
        sentence_feature = self.sentence_genaration(word_feature,sentence_mask,self.device)
        image1 = self.g_W1(image1)
        image1,att_image = self.Encoder2(image1, mask=None,mask1=None,mask2=None)

        global_ouput_1 = self.SectionOne(global_text.view(-1, 4, 768), image1,section_prompt)
        global_ouput_2 = self.SentenceFour(sentence_feature, image1, sentence_prompt)

        u33 = torch.cat([global_ouput_1.unsqueeze(1), global_ouput_2.unsqueeze(1)], dim=1)
        logits = self.classifier(torch.max(u33,dim=1)[0])
        return logits

def main():
    #Set default configuration in args.py
    args = get_args()
    dataset_map = {'AAPD': AAPDProcessor}

    output_modes = {"rte": "classification"}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    args.device = device
    args.n_gpu = n_gpu  # 1
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES  # 12
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL  # True
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    processor = dataset_map[args.dataset]()

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    # rea = Rea()
    MultiHeadedAttention1 = MultiHeadedAttention(16, 768)
    positionembeddings1 = positionembeddings(device)
    PositionwiseFeedForward1 = PositionwiseFeedForward(768, 3072)
    EncoderLayer2 = EncoderLayer(768, MultiHeadedAttention1, PositionwiseFeedForward1, 0.1)
    Encoder2 = Encoder(EncoderLayer2, 1)
    pretrain_model_dir = '/home/ltf/code/data/bert-base-uncased/'
    model = ClassifyModel(pretrain_model_dir, num_labels=args.num_labels,args=args,Encoder2=Encoder2, device=device, is_lock=False)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if not args.trained_model:
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install NVIDIA Apex for FP16 training")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
            scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
                                             warmup_steps=args.warmup_proportion * num_train_optimization_steps)

        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()
        model = torch.load(trainer.snapshot_path)
    else:
        model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=args.num_labels)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

    evaluate_split(model, processor, tokenizer, args, split='dev')
    evaluate_split(model, processor, tokenizer, args, split='test')


if __name__ == "__main__":
    main()
