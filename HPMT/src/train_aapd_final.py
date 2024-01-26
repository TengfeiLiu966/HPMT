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
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_split(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
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
class s2tSA(customizedModule):
    def __init__(self, hidden_size):
        super(s2tSA, self).__init__()

        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, (block_num), seq_len, hidden_size)
        :return: s: (batch, (block_num), hidden_size)
        """

        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g1,g2, h):
        Z = self.drop(h)
        # weights = torch.max(torch.matmul(h,section_feature.transpose(1, 2)),2)[0]
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g1,g2, h, self.k)

# def top_k_graph(scores,h, k):
#     num_nodes = h.shape[1]                                       #一共几个节点   1014
#     values, idx = torch.topk(scores, max(2, k))   #最大值得索引以及值
#     for i in range(4):
#         new_h = h[i, idx[i, :], :]
#     values = torch.unsqueeze(values, -1)                         #新增一个维度
#     new_h = torch.mul(new_h, values)         #                   #新的特征
#     return new_h
def top_k_graph(scores, g1,g2, h, k):
    num_nodes = g1.shape[1]                                       #一共几个节点
    values, idx = torch.topk(scores, max(2, k))                   #最大值得索引以及值
    # new_h = h[:, idx, :]                                        #取出特征
    # for i in range(2):
    #     new_h = h[i, idx[i,:], :]
    new_h = []
    for i in range(4):
        new_h.append(h[i, idx[i, :], :].unsqueeze(0))
    new_h = torch.cat([new_h[0], new_h[1],new_h[2], new_h[3]], 0)                   #组合的新特征
    values = torch.unsqueeze(values, -1)                         #新增一个维度   2*512*738
    new_h = torch.mul(new_h, values)         #                   #新的特征       2*512*1     2*512*768
    #下面是对权重的选择
    g_sentence = []
    g_section = []
    for i in range(4):
        g11 = g1[i,idx[i, :],:]
        g11 = g11[:, idx[i,:]]
        g_section.append(g11.unsqueeze(0))
        g22 = g2[i,idx[i,:],:]
        g22 = g22[:, idx[i,:]]
        g_sentence.append(g22.unsqueeze(0))
    return torch.cat([g_section[0],g_section[1],g_section[2],g_section[3]],0),torch.cat([g_sentence[0],g_sentence[1],g_section[2],g_section[3]],0), new_h

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class Self_att(nn.Module):
    def __init__(self):
        super(Self_att, self).__init__()

        self.f1 = nn.Linear(768, int(768 / 2))
        self.f2 = nn.Linear(int(768 / 2), 1)
        self.tanh = nn.ReLU(inplace=True)    # !!!!!!!!
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # mask = mask.squeeze(1)
        att = self.f1(x)
        att = self.tanh(att)
        att = self.dropout(att)
        att = self.f2(att)

        b = x.size(0)
        n = x.size(1)
        # att = att.view(b, n).masked_fill(mask, -1e9)
        att = att.view(b, n)
        att = F.softmax(att, dim=1).view(b, n, 1)
        return att

# class Rea(nn.Module):
#     def __init__(self):
#         super(Rea, self).__init__()
#
#         self.f1 = nn.Linear(768, 768)
#         self.f2 = nn.Linear(768, 768)
#         self.f3 = nn.Linear(768, 768)
#         self.f4 = nn.Linear(768, 768)
#
#         self.dropout1 = nn.Dropout(0.1)
#
#     def forward(self, x,section_mask_full,sentence_mask_full):                        #所有的mask都为  4*1016*1016
#         att = torch.matmul(self.f1(x), self.f2(x).transpose(1, 2)) / 28               #这是随机初始化的原始权重4*1016*1016
#         #生成一个对角mask,然后softmax处理
#         # dia_att = F.softmax(att.masked_fill(section_mask_full == 0, -1e9), dim=2)     #通过段落mask，mask掉padding的部分
#         #先验知识
#         # prior_att = F.softmax((2*section_mask_full + 4*sentence_mask_full), dim=2)
#         #知识的融合
#         # fusion_att = dia_att * prior_att
#         graph_r = self.dropout1(att)
#         g_x1 = self.f3(torch.matmul(graph_r, x))
#         #把权重矩阵一分为二
#         no_self_att = F.softmax(att.masked_fill(section_mask_full == 1, -1e9), dim=2)  # 通过段落mask，mask掉padding的部分
#         graph_r1 = self.dropout1(no_self_att)
#         g_x2 = self.f4(torch.matmul(graph_r1, g_x1))
#         g_x = g_x1  + g_x2
#         return g_x
class Rea(nn.Module):
    def __init__(self):
        super(Rea, self).__init__()

        self.f1 = nn.Linear(768, 768)
        self.f2 = nn.Linear(768, 768)
        self.f3 = nn.Linear(768, 768)
        self.f4 = nn.Linear(768, 768)

        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x,section_mask_ful):                        #所有的mask都为  4*1016*1016
        att = torch.matmul(self.f1(x), self.f2(x).transpose(1, 2)) / 28               #这是随机初始化的原始权重4*1016*1016
        #生成一个对角mask,然后softmax处理
        # dia_att = F.softmax(att.masked_fill(section_mask_full == 0, -1e9), dim=2)     #通过段落mask，mask掉padding的部分
        #先验知识
        # prior_att = F.softmax((2*section_mask_full + 4*sentence_mask_full), dim=2)
        #知识的融合
        # fusion_att = dia_att * prior_att
        graph_r = self.dropout1(att)
        g_x1 = self.f3(torch.matmul(graph_r, x))
        #把权重矩阵一分为二
        no_self_att = F.softmax(att.masked_fill(section_mask_full == 1, -1e9), dim=2)  # 通过段落mask，mask掉padding的部分
        graph_r1 = self.dropout1(no_self_att)
        g_x2 = self.f4(torch.matmul(graph_r1, g_x1))
        g_x = g_x1  + g_x2
        return g_x
class ClassifyModel(customizedModule):
    def __init__(self, pretrained_model_name_or_path, args, rea,device, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.args = args
        self.pools = nn.ModuleList()
        self.pools.append(Pool(512, 768, 0.1))
        self.pools.append(Pool(256, 768, 0.1))
        self.pools.append(Pool(128, 768, 0.1))
        self.classifier = nn.Linear(768, self.args.num_labels)
        self.init_mBloSA()
        self.s2tSA = s2tSA(768)
        self.layers = rea
        # self.layers = clones(rea, 4)
        self.device = device
        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)
    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

        self.f_W1 = self.customizedLinear(768 * 3, 768, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(768 * 3, 768)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sentence_mask = None, label=None, ):
        all_token_feature, pooled_feature = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False)

        token_feature = all_token_feature[:, 1:-1, :]                     #获取段中所有单词的特征
        token_feature = torch.reshape(token_feature, (-1,4*(self.args.max_seq_length-2), 768))            #batch * word num * feature dim
        #******************************************************************************************************************************************************
        #生成mask
        # h_mask = sentence_mask.unsqueeze(-1).expand(self.args.train_batch_size, 4, self.args.max_seq_length-2, self.args.max_seq_length-2)
        # v_mask = sentence_mask.unsqueeze(-2).expand(self.args.train_batch_size, 4, self.args.max_seq_length-2, self.args.max_seq_length-2)
        # sentence_mask_full = torch.tensor([scipy.linalg.block_diag(i[0,:], i[1,:], i[2,:], i[3,:]) for i in (h_mask == v_mask).cpu()],dtype=torch.float).to(self.device)
        #
        # attention_mask_1 = torch.tensor(attention_mask.view(-1, 4, self.args.max_seq_length).unsqueeze(-1),dtype = torch.float)
        # attention_mask_2 = torch.matmul(attention_mask_1, attention_mask_1.transpose(2, 3))[:, :, 1:-1, 1:-1]
        # section_mask_full = torch.tensor([scipy.linalg.block_diag(i[0,:], i[1,:], i[2,:], i[3,:]) for i in attention_mask_2.cpu()],dtype=torch.float).to(self.device)   #2*1016*1016
        #*************************************************************************************************************************************************
        # #解耦图池化模块*******
        # attention_output_18 = self.layers(token_feature,sentence_mask_full,section_mask_full)
        # #残差控制模块
        # # G = F.sigmoid(self.g_W1(token_feature) + self.g_W2(attention_output_18) + self.g_b)
        # # ret = G * token_feature + (1 - G) * attention_output_18
        # #图池化
        # section_mask,sentence_mask,new_att = self.pools[0](section_mask_full,sentence_mask_full,attention_output_18)  # 2*14*768
        #
        # attention_output_16 = self.layers(new_att,section_mask,sentence_mask)
        # section_mask,sentence_mask,new_att = self.pools[1](section_mask,sentence_mask,attention_output_16)
        #
        # attention_output_14 = self.layers(new_att,section_mask,sentence_mask)
        # section_mask,sentence_mask,new_att = self.pools[2](section_mask,sentence_mask,attention_output_14)
        #
        # attention_output_12 = self.layers(new_att,section_mask,sentence_mask)
        # 解耦图池化模块*******
        attention_output_18 = self.layers(token_feature,section_mask_full)
        # 残差控制模块
        # G = F.sigmoid(self.g_W1(token_feature) + self.g_W2(attention_output_18) + self.g_b)
        # ret = G * token_feature + (1 - G) * attention_output_18
        # 图池化
        section_mask, sentence_mask, new_att = self.pools[0](attention_output_18)  # 2*14*768

        attention_output_16 = self.layers(new_att)
        section_mask, sentence_mask, new_att = self.pools[1](attention_output_16)

        attention_output_14 = self.layers(new_att)
        section_mask, sentence_mask, new_att = self.pools[2](attention_output_14)

        attention_output_12 = self.layers(new_att)

        u = torch.cat([torch.max(attention_output_18, dim=1)[0].unsqueeze(1),
                       torch.max(attention_output_12, dim=1)[0].unsqueeze(1),
                       torch.max(attention_output_16, dim=1)[0].unsqueeze(1),
                       torch.max(attention_output_14, dim=1)[0].unsqueeze(1),
                       torch.max(pooled_feature.view(-1, 4, 768), dim=1)[0].unsqueeze(1),], dim=1)

        # # fusion = self.f_W1(torch.cat([pooled.view(-1, 20, 768), attention_output], dim=2))
        # # G = F.sigmoid(self.f_W2(torch.cat([pooled.view(-1, 20, 768), attention_output], dim=2)))
        # # u = G * fusion + (1 - G) * attention_output
        # fusion = self.f_W1(u)
        # G = F.sigmoid(self.f_W2(u))
        # (batch, n, word_dim)
        # u = G * fusion + (1 - G) * torch.mean(attention_output_10, dim=1)
        logits = self.s2tSA(u)
        # logits = torch.max(new_att, dim=1)[0]
        logits = self.classifier(logits)
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
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES  # 54
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
    rea = Rea()

    # pretrain_model_dir = '/home/ltf/code/data/bert-base-uncased/'
    pretrain_model_dir ='/home/ltf/code/data/scibert_scivocab_uncased/'
    # model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=args.num_labels)
    model = ClassifyModel(pretrain_model_dir, args=args, rea = rea,device=device, is_lock=False)
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
# CUDA_VISIBLE_DEVICES=1,2 python -u train_Yahoo_fine_tune_Bert_zeroshot.py --task_name rte --do_train --do_lower_case --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --data_dir '' --output_dir ''
