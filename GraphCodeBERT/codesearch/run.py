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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from collections import defaultdict
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing
cpu_cont = 16
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA execution for debugging

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,
                 ori2cur_pos,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.ori2cur_pos=ori2cur_pos
        
        
def convert_examples_to_features(item):
    js,tokenizer,args=item
    #code
    parser=parsers[args.lang]
    #extract data flow
    code_tokens,dfg=extract_dataflow(js['original_string'],parser,args.lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'],ori2cur_pos=ori2cur_pos)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            self.examples = []
            data=[]
            with open(file_path) as f:
                for line in f:
                    line=line.strip()
                    js=json.loads(line)
                    data.append((js,tokenizer,args))
            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids)))) 
                logger.info("ori2cur_pos: {}".format(example.ori2cur_pos))          
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                            self.args.code_length+self.args.data_flow_length),dtype=np.bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
        ori2cur_pos_list = [[start, end] for start, end in self.examples[item].ori2cur_pos.values()]
        # 设置填充长度
        max_len = self.args.code_length
        ori2cur_pos_list_padded = ori2cur_pos_list + [[0, 0]] * (max_len - len(ori2cur_pos_list))
        if len(ori2cur_pos_list_padded) > max_len:
            ori2cur_pos_list_padded = ori2cur_pos_list_padded[:max_len]
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].nl_ids),
              torch.tensor(ori2cur_pos_list_padded))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)[1]
            nl_vec = model(nl_inputs=nl_inputs)[1]
            
            #calculate scores and loss
            scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

        # output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx+1))
        # ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')

        # model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
        # model.to(args.device)
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

        # output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx + 1))
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # model_to_save = model.module if hasattr(model, 'module') else model
        # ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')
        # logger.info("Saving model checkpoint to %s", ckpt_output_path)
        # torch.save(model_to_save.state_dict(), ckpt_output_path)

def save_attention_features(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    input_path = "/home/yiming/cophi/training_dynamic/NL-code-search-Adv/model/codebert/token_alignment/label_human.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')  # 去除行末的逗号
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])
    
    sample_index = idx_list
    sample_align = match_list
    
    model.eval()
    tr_num,tr_loss,best_mrr=0,0,0 
    # for idx in range(args.num_train_epochs): 
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(51))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_new_batch_retrieval.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    code_token, nl_token = [], []
    code_tokens, nl_tokens = [], []
    all_code_cls_attention = []
    all_nl_cls_attention = []
    all_code_attention = []
    all_nl_attention = []
    code_tokens_strs = []
    tokenized_id = []

    for step,batch in enumerate(train_dataloader):
        #get inputs
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size

        for i in range(len(sample_index)):
            if batch_start_idx <= sample_index[i] < batch_end_idx:
                code_inputs = batch[0].to(args.device)  
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                nl_inputs = batch[3].to(args.device)
                code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
                nl_outputs = model(nl_inputs=nl_inputs)
                local_index = sample_index[i] - batch_start_idx
                print(local_index)

                ori2cur_pos = batch[4].to(args.device)
                tokenized_id.append(ori2cur_pos[local_index].tolist())

                code_token_ids = code_inputs[local_index].tolist()
                code_tokens_str = [tokenizer.convert_ids_to_tokens(id) for id in code_token_ids]
                code_tokens_strs.append(code_tokens_str)

                code_last_hidden_state = code_outputs[0]
                code_tokens.append(code_last_hidden_state[local_index].unsqueeze(0).cpu().detach().numpy())
                nl_last_hidden_state = nl_outputs[0]
                nl_tokens.append(nl_last_hidden_state[local_index].unsqueeze(0).cpu().detach().numpy())

                # get code and nl vectors
                code_vec = code_outputs[1]
                nl_vec = nl_outputs[1]

                # scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
                # loss_fct = CrossEntropyLoss(reduction='none')
                # loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
                # eval_loss.extend(loss.detach().cpu().numpy())

                # 获取注意力权重
                code_attentions = code_outputs[2]  # List of tensors, one for each layer
                nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

                code_last_layer_attention = code_attentions[-1]
                code_cls_attention = code_last_layer_attention[local_index, :, 0, :].mean(dim=0)
                
                nl_last_layer_attention = nl_attentions[-1]
                nl_cls_attention = nl_last_layer_attention[local_index, :, 0, :].mean(dim=0)

                # # 对多头 attention 取平均值
                # mean_code_attention = code_last_layer_attention.mean(dim=1)  # 形状为 [batch_size, seq_len, seq_len]
                # mean_nl_attention = nl_last_layer_attention.mean(dim=1) 

                # for layer_idx in range(len(code_attentions)-1, -1, -1):
                #     # attention 的形状: (batch_size, num_heads, seq_length, seq_length)
                #     code_attention = code_attentions[layer_idx].mean(dim=1)  # (batch_size, seq_length, seq_length)
                #     nl_attention = nl_attentions[layer_idx].mean(dim=1)
                    
                #     # 计算每个 token 对 CLS token 的贡献
                #     if layer_idx == len(code_attentions)-1:
                #         code_cls_attention = code_attention[:, 0, :]  # (batch_size, seq_length)
                #         nl_cls_attention = nl_attention[:, 0, :]
                #     else:
                #         code_cls_attention = torch.matmul(code_cls_attention.unsqueeze(1), code_attention).squeeze(1)  # (batch_size, seq_length)
                #         nl_cls_attention = torch.matmul(nl_cls_attention.unsqueeze(1), nl_attention).squeeze(1)

                for layer_idx in range(len(code_attentions) - 1, -1, -1):
                    # attention 的形状: (batch_size, num_heads, seq_length, seq_length)
                    code_attention = code_attentions[layer_idx][local_index].mean(dim=0)  # (batch_size, seq_length, seq_length)
                    nl_attention = nl_attentions[layer_idx][local_index].mean(dim=0)      # (batch_size, seq_length, seq_length)

                    # 如果是最后一层，直接初始化累乘结果
                    if layer_idx == len(code_attentions) - 1:
                        # 对于最后一层，初始化所有 token 的累乘结果
                        mean_code_attention = code_attention  # (batch_size, seq_length, seq_length)
                        mean_nl_attention = nl_attention      # (batch_size, seq_length, seq_length)
                    else:
                        # 对于其他层，将累乘结果和当前层的 attention 进行矩阵乘法
                        mean_code_attention = torch.matmul(mean_code_attention, code_attention)  # (batch_size, seq_length, seq_length)
                        mean_nl_attention = torch.matmul(mean_nl_attention, nl_attention) 

                all_code_cls_attention.append(code_cls_attention.unsqueeze(0).cpu().detach().numpy())
                all_nl_cls_attention.append(nl_cls_attention.unsqueeze(0).cpu().detach().numpy())

                all_code_attention.append(mean_code_attention.unsqueeze(0).cpu().detach().numpy())
                all_nl_attention.append(mean_nl_attention.unsqueeze(0).cpu().detach().numpy())

                # code_token.append(code_vec[local_index].unsqueeze(0).cpu().detach().numpy())
                # nl_token.append(nl_vec[local_index].unsqueeze(0).cpu().detach().numpy())

                code_token.append(code_vec.cpu().detach().numpy())
                nl_token.append(nl_vec.cpu().detach().numpy())

                # code_tokens.append(code_outputs[0].cpu().detach().numpy())
                # nl_tokens.append(nl_outputs[0].cpu().detach().numpy())
        
    code_token = np.concatenate(code_token, 0)
    nl_token = np.concatenate(nl_token, 0)
    code_tokens = np.concatenate(code_tokens, 0)
    nl_tokens = np.concatenate(nl_tokens, 0)
    all_code_cls_attention = np.concatenate(all_code_cls_attention, 0)
    all_nl_cls_attention = np.concatenate(all_nl_cls_attention, 0)

    output_dir = "/home/yiming/cophi/training_dynamic/gcb_tokens_temp"
    
    # print(code_token.shape, nl_token.shape)
    code_token_output_path = os.path.join(output_dir, 'new_train_code_cls_token_new_batch_retrieval.npy')
    nl_token_output_path = os.path.join(output_dir, 'new_train_nl_cls_token_new_batch_retrieval.npy')
    np.save(code_token_output_path, code_token)
    np.save(nl_token_output_path, nl_token)

    ### cls accumulative attention
    print(all_nl_cls_attention.shape, all_code_cls_attention.shape)
    code_attention_output_path = os.path.join(output_dir, 'new_train_code_attention_new_batch_retrieval.npy')
    nl_attention_output_path = os.path.join(output_dir, 'new_train_nl_attention_new_batch_retrieval.npy')
    np.save(code_attention_output_path, all_code_cls_attention)
    np.save(nl_attention_output_path, all_nl_cls_attention)

    print(code_tokens.shape, nl_tokens.shape)
    code_token_output_path = os.path.join(output_dir, 'new_train_code_tokens_new_batch_retrieval.npy')
    nl_tokens_output_path = os.path.join(output_dir, 'new_train_nl_tokens_new_batch_retrieval.npy')
    np.save(code_token_output_path, code_tokens)
    np.save(nl_tokens_output_path, nl_tokens)

    ### all token attention
    # print(all_nl_attention.shape, all_code_attention.shape)
    code_attention_output_path = os.path.join(output_dir, 'new_train_all_code_attention_new_batch_retrieval.npy')
    nl_attention_output_path = os.path.join(output_dir, 'new_train_all_nl_attention_new_batch_retrieval.npy')
    np.save(code_attention_output_path, all_code_attention)
    np.save(nl_attention_output_path, all_nl_attention)

    with open('/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_code_tokens.json', 'w') as f:
        json.dump(code_tokens_strs, f)

    with open('/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1/tokenized_id_black.json', 'w') as f:
        json.dump(tokenized_id, f)

    # ### eval loss
    # eval_loss = np.array(eval_loss)
    # print(eval_loss.shape)
    # eval_loss_path = os.path.join(output_dir, 'new_train_loss.npy')
    # np.save(eval_loss_path, eval_loss)

def save_train_features(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(3))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_retrieval_only.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    nl_token = []
    nl_tokens = []
    all_nl_cls_attention = []
    code_token = []
    code_tokens = []
    all_code_cls_attention = []
    tokenized_id = []

    # Load sample indices from file
    input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_sorted.jsonl"
    sample_indices = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')  # 去除行末的逗号
            json_obj = json.loads(line)
            sample_indices.append(json_obj['idx'])

    for step,batch in enumerate(train_dataloader):
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size

        # Check if current batch contains any samples we want
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        if not any(idx in sample_indices for idx in batch_indices):
            continue

        # Create mask for samples we want to keep
        batch_mask = torch.tensor([idx in sample_indices for idx in batch_indices]).to(args.device)

        nl_inputs = batch[3].to(args.device)
        nl_outputs = model(nl_inputs=nl_inputs)

        code_inputs = batch[0].to(args.device)  
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)

        # Only store features for samples in our index list
        nl_vec = nl_outputs[1][batch_mask]
        nl_token.append(nl_vec.cpu().detach().numpy())

        nl_attentions = nl_outputs[3]
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[:, :, 0, :].mean(dim=1)[batch_mask]
        all_nl_cls_attention.append(nl_cls_attention.cpu().detach().numpy())

        code_vec = code_outputs[1][batch_mask]
        code_token.append(code_vec.cpu().detach().numpy())

        code_attentions = code_outputs[3]
        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[:, :, 0, :].mean(dim=1)[batch_mask]
        all_code_cls_attention.append(code_cls_attention.cpu().detach().numpy())

        nl_last_hidden_state = nl_outputs[2][-2]  # Get -2 layer hidden states
        nl_final_hidden_state = nl_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        filtered_nl_hidden_states = nl_last_hidden_state[batch_mask]
        filtered_nl_final_states = nl_final_hidden_state[batch_mask]
        filtered_nl_hidden_states[:,0,:] = filtered_nl_final_states
        nl_tokens.append(filtered_nl_hidden_states.cpu().detach().numpy())

        code_last_hidden_state = code_outputs[2][-2]  # Get -2 layer hidden states
        code_final_hidden_state = code_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        filtered_code_hidden_states = code_last_hidden_state[batch_mask]
        filtered_code_final_states = code_final_hidden_state[batch_mask]
        filtered_code_hidden_states[:,0,:] = filtered_code_final_states
        code_tokens.append(filtered_code_hidden_states.cpu().detach().numpy())

        # # save code tokenize result
        # ori2cur_pos = batch[4].to(args.device)
        # # 使用 extend 将整个 ori2cur_pos 列表的内容添加到 tokenized_id
        # tokenized_id.extend(ori2cur_pos.tolist())


    output_dir = "/home/yiming/cophi/training_dynamic/features/retri_autolabel"

    nl_token = np.concatenate(nl_token, 0)
    nl_tokens = np.concatenate(nl_tokens, 0)
    all_nl_cls_attention = np.concatenate(all_nl_cls_attention, 0)
    
    print(nl_token.shape)
    nl_token_output_path = os.path.join(output_dir, 'train_nl_cls_token_retri.npy')
    np.save(nl_token_output_path, nl_token)

    ## cls accumulative attention
    print(all_nl_cls_attention.shape)
    nl_attention_output_path = os.path.join(output_dir, 'train_nl_attention_retri.npy')
    np.save(nl_attention_output_path, all_nl_cls_attention)

    print(nl_tokens.shape)
    nl_tokens_output_path = os.path.join(output_dir, 'train_nl_tokens_retri.npy')
    np.save(nl_tokens_output_path, nl_tokens)
    
    # print(len(tokenized_id))
    # with open('/home/yiming/cophi/training_dynamic/features/tokenized_id_train.json', 'w') as f:
    #     json.dump(tokenized_id, f)

    code_token = np.concatenate(code_token, 0)
    code_tokens = np.concatenate(code_tokens, 0)
    all_code_cls_attention = np.concatenate(all_code_cls_attention, 0)
    
    print(code_token.shape)
    code_token_output_path = os.path.join(output_dir, 'train_code_cls_token_retri.npy')
    np.save(code_token_output_path, code_token)

    print(all_code_cls_attention.shape)
    code_attention_output_path = os.path.join(output_dir, 'train_code_attention_retri.npy')
    np.save(code_attention_output_path, all_code_cls_attention)

    print(code_tokens.shape)
    code_token_output_path = os.path.join(output_dir, 'train_code_tokens_retri.npy')
    np.save(code_token_output_path, code_tokens)


def save_valid_attention_features(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.eval_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(40))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_new_auto_label_aa_one_sample.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    
    nl_token = []  # For CLS tokens
    nl_tokens = []  # For selected samples' hidden states
    all_nl_cls_attention = []  # For attention features

    for step,batch in enumerate(train_dataloader):
        
        nl_inputs = batch[3].to(args.device)
        nl_outputs = model(nl_inputs=nl_inputs)
        
        # Always store CLS tokens and attention features
        nl_vec = nl_outputs[1]
        nl_token.append(nl_vec.cpu().detach().numpy())
        
        nl_attentions = nl_outputs[3]
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[:, :, 0, :].mean(dim=1)
        all_nl_cls_attention.append(nl_cls_attention.cpu().detach().numpy())

        nl_last_hidden_state = nl_outputs[2][-2]  # Get -2 layer hidden states
        nl_final_hidden_state = nl_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        # Replace the first token of -2 layer with first token of -1 layer
        nl_last_hidden_state[:,0,:] = nl_final_hidden_state
        nl_tokens.append(nl_last_hidden_state.cpu().detach().numpy())

    output_dir = "/home/yiming/cophi/training_dynamic/features/aa_possim"
    
    # Save CLS tokens
    nl_token = np.concatenate(nl_token, 0)
    print(nl_token.shape)
    nl_token_output_path = os.path.join(output_dir, 'valid_nl_cls_token_aa.npy')
    np.save(nl_token_output_path, nl_token)

    # Save attention features
    all_nl_cls_attention = np.concatenate(all_nl_cls_attention, 0)
    print(all_nl_cls_attention.shape)
    nl_attention_output_path = os.path.join(output_dir, 'valid_nl_attention_aa.npy')
    np.save(nl_attention_output_path, all_nl_cls_attention)

def save_valid_attention_features_nl(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.eval_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(40))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_retrieval_only.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    
    nl_tokens = []  # For selected samples' hidden states

    sample_indices = [3112]

    for step,batch in enumerate(train_dataloader):
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size

        # Check if current batch contains any samples we want
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        if not any(idx in sample_indices for idx in batch_indices):
            continue

        nl_inputs = batch[3].to(args.device)
        nl_outputs = model(nl_inputs=nl_inputs)
        # Only store hidden states for samples in our index list
        nl_last_hidden_state = nl_outputs[2][-2]  # Get -2 layer hidden states
        nl_final_hidden_state = nl_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        
        batch_mask = torch.tensor([idx in sample_indices for idx in batch_indices]).to(args.device)
        filtered_nl_hidden_states = nl_last_hidden_state[batch_mask]
        filtered_nl_final_states = nl_final_hidden_state[batch_mask]
        
        if len(filtered_nl_hidden_states) > 0:
            # Replace the first token of -2 layer with first token of -1 layer
            filtered_nl_hidden_states[:,0,:] = filtered_nl_final_states
            nl_tokens.append(filtered_nl_hidden_states.cpu().detach().numpy())

        
    output_dir = "/home/yiming/cophi/training_dynamic/features/retri_single_sample"

    if nl_tokens:
        nl_tokens = np.concatenate(nl_tokens, 0)
        print(nl_tokens.shape)
        nl_tokens_output_path = os.path.join(output_dir, 'valid_nl_tokens_retrieval.npy')
        np.save(nl_tokens_output_path, nl_tokens)

def save_valid_attention_features_nl_aa(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.eval_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(40))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_new_auto_label_aa_one_sample.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    
    nl_tokens = []  # For selected samples' hidden states

    sample_indices = [3112]

    for step,batch in enumerate(train_dataloader):
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size

        # Check if current batch contains any samples we want
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        if not any(idx in sample_indices for idx in batch_indices):
            continue

        nl_inputs = batch[3].to(args.device)
        nl_outputs = model(nl_inputs=nl_inputs)
        # Only store hidden states for samples in our index list
        nl_last_hidden_state = nl_outputs[2][-2]  # Get -2 layer hidden states
        nl_final_hidden_state = nl_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        
        batch_mask = torch.tensor([idx in sample_indices for idx in batch_indices]).to(args.device)
        filtered_nl_hidden_states = nl_last_hidden_state[batch_mask]
        filtered_nl_final_states = nl_final_hidden_state[batch_mask]
        
        if len(filtered_nl_hidden_states) > 0:
            # Replace the first token of -2 layer with first token of -1 layer
            filtered_nl_hidden_states[:,0,:] = filtered_nl_final_states
            nl_tokens.append(filtered_nl_hidden_states.cpu().detach().numpy())

        
    output_dir = "/home/yiming/cophi/training_dynamic/features/aa_possim"

    if nl_tokens:
        nl_tokens = np.concatenate(nl_tokens, 0)
        print(nl_tokens.shape)
        nl_tokens_output_path = os.path.join(output_dir, 'valid_nl_tokens_aa.npy')
        np.save(nl_tokens_output_path, nl_tokens)

def save_valid_code_attention_features(args, model, tokenizer,pool):
    torch.cuda.empty_cache()
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.codebase_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(40))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_retrieval_only.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False)
    model.to(args.device)
    
    code_token = []  # For CLS tokens
    code_tokens = []  # For selected samples' hidden states  
    all_code_cls_attention = []  # For attention features

    # Load sample indices from file
    sample_indices = [36111]

    for step,batch in enumerate(train_dataloader):
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size

        # Check if current batch contains any samples we want
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        if not any(idx in sample_indices for idx in batch_indices):
            continue
            
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        
        code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)

        
        # # Always store CLS tokens and attention features
        # code_vec = code_outputs[1]
        # code_token.append(code_vec.cpu().detach().numpy())
        
        # code_attentions = code_outputs[3]
        # code_last_layer_attention = code_attentions[-1]
        # code_cls_attention = code_last_layer_attention[:, :, 0, :].mean(dim=1)
        # all_code_cls_attention.append(code_cls_attention.cpu().detach().numpy())


        # Only store hidden states for samples in our index list
        code_last_hidden_state = code_outputs[2][-2]  # Get -2 layer hidden states
        code_final_hidden_state = code_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        
        batch_mask = torch.tensor([idx in sample_indices for idx in batch_indices]).to(args.device)
        filtered_code_hidden_states = code_last_hidden_state[batch_mask]
        filtered_code_final_states = code_final_hidden_state[batch_mask]
        
        if len(filtered_code_hidden_states) > 0:
            # Replace the first token of -2 layer with first token of -1 layer
            filtered_code_hidden_states[:,0,:] = filtered_code_final_states
            code_tokens.append(filtered_code_hidden_states.cpu().detach().numpy())

        
    output_dir = "/home/yiming/cophi/training_dynamic/features/retri_single_sample"
    
    # # Save CLS tokens
    # code_token = np.concatenate(code_token, 0)
    # print(code_token.shape)
    # code_token_output_path = os.path.join(output_dir, 'valid_code_cls_token_retrieval.npy')
    # np.save(code_token_output_path, code_token)

    # # Save attention features  
    # all_code_cls_attention = np.concatenate(all_code_cls_attention, 0)
    # print(all_code_cls_attention.shape)
    # code_attention_output_path = os.path.join(output_dir, 'valid_code_attention_retrieval.npy')
    # np.save(code_attention_output_path, all_code_cls_attention)

    # Save selected samples' hidden states
    if code_tokens:
        code_tokens = np.concatenate(code_tokens, 0)
        print(code_tokens.shape)
        code_tokens_output_path = os.path.join(output_dir, 'valid_code_tokens_retrieval.npy')
        np.save(code_tokens_output_path, code_tokens)


def save_valid_code_attention_features_aa(args, model, tokenizer,pool):
    torch.cuda.empty_cache()
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.codebase_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(40))
    ckpt_output_path = os.path.join(output_dir, 'subject_model_new_auto_label_aa_one_sample.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False)
    model.to(args.device)
    
    code_token = []  # For CLS tokens
    code_tokens = []  # For selected samples' hidden states  
    all_code_cls_attention = []  # For attention features

    # Load sample indices from file
    sample_indices = [36111]

    for step,batch in enumerate(train_dataloader):
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size

        # Check if current batch contains any samples we want
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        if not any(idx in sample_indices for idx in batch_indices):
            continue
            
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
        
        # # Always store CLS tokens and attention features
        # code_vec = code_outputs[1]
        # code_token.append(code_vec.cpu().detach().numpy())
        
        # code_attentions = code_outputs[3]
        # code_last_layer_attention = code_attentions[-1]
        # code_cls_attention = code_last_layer_attention[:, :, 0, :].mean(dim=1)
        # all_code_cls_attention.append(code_cls_attention.cpu().detach().numpy())

        # Only store hidden states for samples in our index list
        code_last_hidden_state = code_outputs[2][-2]  # Get -2 layer hidden states
        code_final_hidden_state = code_outputs[2][-1][:,0,:]  # Get -1 layer hidden states
        
        batch_mask = torch.tensor([idx in sample_indices for idx in batch_indices]).to(args.device)
        filtered_code_hidden_states = code_last_hidden_state[batch_mask]
        filtered_code_final_states = code_final_hidden_state[batch_mask]
        
        if len(filtered_code_hidden_states) > 0:
            # Replace the first token of -2 layer with first token of -1 layer
            filtered_code_hidden_states[:,0,:] = filtered_code_final_states
            code_tokens.append(filtered_code_hidden_states.cpu().detach().numpy())

    output_dir = "/home/yiming/cophi/training_dynamic/features/aa_possim"
    
    # # Save CLS tokens
    # code_token = np.concatenate(code_token, 0)
    # print(code_token.shape)
    # code_token_output_path = os.path.join(output_dir, 'valid_code_cls_token_aa.npy')
    # np.save(code_token_output_path, code_token)

    # # Save attention features  
    # all_code_cls_attention = np.concatenate(all_code_cls_attention, 0)
    # print(all_code_cls_attention.shape)
    # code_attention_output_path = os.path.join(output_dir, 'valid_code_attention_aa.npy')
    # np.save(code_attention_output_path, all_code_cls_attention)

    # Save selected samples' hidden states
    if code_tokens:
        code_tokens = np.concatenate(code_tokens, 0)
        print(code_tokens.shape)
        code_tokens_output_path = os.path.join(output_dir, 'valid_code_tokens_aa.npy')
        np.save(code_tokens_output_path, code_tokens)

    
def save_tokens_features(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.eval()
    # output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(1))
    # ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')

    # model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    code_token, nl_token = [], []

    for step,batch in enumerate(train_dataloader):
        if step % 1000 == 0:
            logger.info("  cur steps = %d", step)
        code_inputs = batch[0].to(args.device)  
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        # nl_inputs = batch[3].to(args.device)
        code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
        # nl_outputs = model(nl_inputs=nl_inputs)

        #get code and nl vectors
        code_vec = code_outputs[1]
        # nl_vec = nl_outputs[1]

        code_token.append(code_vec.cpu().detach().numpy())
        # nl_token.append(nl_vec.cpu().detach().numpy())
        
    code_token = np.concatenate(code_token, 0)
    # nl_token = np.concatenate(nl_token, 0)

    output_dir = "/home/yiming/cophi/training_dynamic/gcb_tokens_temp"

    print(code_token.shape)
    code_token_output_path = os.path.join(output_dir, 'train_code_cls_token_pt.npy')
    # nl_token_output_path = os.path.join(output_dir, 'train_nl_cls_token_pt.npy')
    np.save(code_token_output_path, code_token)

def save_attention_features_code(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.codebase_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    # for idx in range(args.num_train_epochs): 
    output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(51))
    ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')

    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(args.device)
    code_token, nl_token = [], []
    code_tokens, nl_tokens = [], []
    all_code_cls_attention = []
    all_nl_cls_attention = []
    eval_loss = []
    for step,batch in enumerate(train_dataloader):
        #get inputs
        code_inputs = batch[0].to(args.device)  
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)

        code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
        nl_outputs = model(nl_inputs=nl_inputs)

        #get code and nl vectors
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]

        # 获取注意力权重
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        for layer_idx in range(len(code_attentions)-1, -1, -1):
            # attention 的形状: (batch_size, num_heads, seq_length, seq_length)
            code_attention = code_attentions[layer_idx].mean(dim=1)  # (batch_size, seq_length, seq_length)
            nl_attention = nl_attentions[layer_idx].mean(dim=1)
            
            # 计算每个 token 对 CLS token 的贡献
            if layer_idx == len(code_attentions)-1:
                code_cls_attention = code_attention[:, 0, :]  # (batch_size, seq_length)
                nl_cls_attention = nl_attention[:, 0, :]
            else:
                code_cls_attention = torch.matmul(code_cls_attention.unsqueeze(1), code_attention).squeeze(1)  # (batch_size, seq_length)
                nl_cls_attention = torch.matmul(nl_cls_attention.unsqueeze(1), nl_attention).squeeze(1)

        all_code_cls_attention.append(code_cls_attention.cpu().detach().numpy())
        all_nl_cls_attention.append(nl_cls_attention.cpu().detach().numpy())

        code_token.append(code_vec.cpu().detach().numpy())
        nl_token.append(nl_vec.cpu().detach().numpy())
        
        code_tokens.append(code_outputs[0].cpu().detach().numpy())
        # nl_tokens.append(nl_outputs[0].cpu().detach().numpy())

    code_token = np.concatenate(code_token, 0)
    nl_token = np.concatenate(nl_token, 0)
    code_tokens = np.concatenate(code_tokens, 0)
    # nl_tokens = np.concatenate(nl_tokens, 0)
    all_code_cls_attention = np.concatenate(all_code_cls_attention, 0)
    all_nl_cls_attention = np.concatenate(all_nl_cls_attention, 0)

    print(code_token.shape, nl_token.shape)
    code_token_output_path = os.path.join(output_dir, 'valid_code_cls_token_overfit_2.npy')
    # nl_token_output_path = os.path.join(output_dir, 'new_valid_nl_cls_token.npy')
    np.save(code_token_output_path, code_token)
    # np.save(nl_token_output_path, nl_token)

    ### cls accumulative attention
    print(all_nl_cls_attention.shape, all_code_cls_attention.shape)
    code_attention_output_path = os.path.join(output_dir, 'valid_code_accumulative_attention_overfit_2.npy')
    # nl_attention_output_path = os.path.join(output_dir, 'new_valid_nl_accumulative_attention.npy')
    np.save(code_attention_output_path, all_code_cls_attention)
    # np.save(nl_attention_output_path, all_nl_cls_attention)

    print(code_tokens.shape)
    code_tokens_output_path = os.path.join(output_dir, 'valid_code_tokens_2.npy')
    # nl_tokens_output_path = os.path.join(output_dir, 'new_valid_nl_tokens_overfit.npy')
    np.save(code_tokens_output_path, code_tokens)
    # np.save(nl_tokens_output_path, nl_tokens)


def train_alignment(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    input_path = "/home/yiming/cophi/training_dynamic/NL-code-search-Adv/model/codebert/token_alignment/label0812_align.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')  # 去除行末的逗号
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])
    
    sample_index = idx_list
    sample_align = match_list
    
    model.train()

    tr_num,tr_loss,best_mrr=0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)

            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size

            code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            nl_outputs = model(nl_inputs=nl_inputs)

            computed_alignment_loss = False
            for i in range(len(sample_index)):
                if batch_start_idx <= sample_index[i] < batch_end_idx:
                    local_index = sample_index[i] - batch_start_idx
                    if isinstance(model, torch.nn.DataParallel):
                        loss, all_loss, code_vec, nl_vec, align_loss, noisy_attention_loss = model.module.alignment(code_inputs, code_outputs, nl_outputs, local_index, sample_align[i], args)
                    else:
                        loss, all_loss, code_vec, nl_vec, align_loss, noisy_attention_loss = model.alignment(code_inputs, code_outputs, nl_outputs, local_index, sample_align[i], args)
                    
                    if all_loss is not None:
                        computed_alignment_loss = True

            if not computed_alignment_loss:
                if isinstance(model, torch.nn.DataParallel):
                    loss = model.module.ori_loss(code_inputs, code_outputs, nl_outputs)
                    all_loss = loss
                else:
                    loss = model.ori_loss(code_inputs, code_outputs, nl_outputs)
                    all_loss = loss
                    
            all_loss = all_loss.mean()
            
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            all_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

            # # 每10个step保存一次模型
            # if step % 10 == 9:  # 0-based index, so 9, 19, 29, ...
            #     output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(step + 1))
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            #     model_to_save = model.module if hasattr(model, 'module') else model
            #     ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')
            #     logger.info("Saving model checkpoint to %s", ckpt_output_path)
            #     torch.save(model_to_save.state_dict(), ckpt_output_path)

        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

# 定义一个 Lambda 函数，使得第一个 epoch 学习率为 0，后面为 args.learning_rate
def lambda_lr(epoch):
    if epoch == 0:
        return 0.0  # 第一个 epoch 的学习率为 0
    else:
        return 1.0  # 后续的 epoch 使用 args.learning_rate

def train_data_preprocess(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,num_training_steps=100*args.num_train_epochs)
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.eval()
    all_tokenized_tokens = []
    for step, batch in enumerate(train_dataloader):
        code_inputs = batch[0].to(args.device)  
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)
        ori2cur_pos = batch[4].to(args.device)
        ## preprocess data to obtain the tokenized code tokens

        code_tokens = code_inputs.tolist()
        code_tokens_str = [tokenizer.convert_ids_to_tokens(id) for id in code_tokens]
        all_tokenized_tokens.extend(code_tokens_str)

    with open('/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1/tokenized_code_tokens.json', 'w') as f:
        json.dump(all_tokenized_tokens, f)


def train_alignment_sample(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    input_path = "/home/yiming/cophi/training_dynamic/NL-code-search-Adv/model/codebert/token_alignment/label_human.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')  # 去除行末的逗号
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])
    
    sample_index = idx_list
    sample_align = match_list
    
    model.eval()
    labelled_mapped_tokens = []
    common_mapped_tokens = []
    tokenized_id = []
    # common_mapped_code_tokens = []
    for step, batch in enumerate(train_dataloader):
        batch_start_idx = step * args.train_batch_size
        batch_end_idx = batch_start_idx + args.train_batch_size
        for i in range(len(sample_index)):
            if batch_start_idx <= sample_index[i] < batch_end_idx:
                local_index = sample_index[i] - batch_start_idx
                code_inputs = batch[0].to(args.device)  
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                nl_inputs = batch[3].to(args.device)
                ori2cur_pos = batch[4].to(args.device)
                ## find common tokens
                code_ids = code_inputs[local_index]
                # print(ori2cur_pos[local_index])
                code_tokens_list = extract_code_tokens_from_alignment(sample_align[i])
                # print("Extracted code tokens:", code_tokens_list)

                # mapped_tokens, mapped_ids = extract_tokens_from_intervals(code_ids, code_tokens_list, ori2cur_pos[local_index])
                mapped_tokens, mapped_ids = code_tokens_list, code_ids
                tokenized_id.append(ori2cur_pos[local_index].tolist())
                # print("Extracted mapped tokens:", mapped_tokens)
                labelled_mapped_tokens.append(mapped_ids)
                # print("Extracted mapped ids:", mapped_ids)

                # 查找在多行中出现的 tokens
                selected_tokens = find_tokens_with_first_occurrence(mapped_tokens)
                # print("selected_tokens:", selected_tokens)
                
                # 找出common tokens对应的原本index
                token_ids = []
                for token_id, info in selected_tokens.items():
                    row = list(info['first_occurrence'].keys())[0]
                    ind = list(info['first_occurrence'].values())[0]
                    token_ids.append(mapped_ids[row][ind])

                code_tokens = code_inputs[local_index].tolist()
                code_tokens_str = [tokenizer.convert_ids_to_tokens(id) for id in code_tokens]
                # 假设 code_tokens_str 是包含所有 token 的列表
                if '</s>' in code_tokens_str:
                    idx = code_tokens_str.index('</s>')
                    truncated_tokens = code_tokens_str[:idx]  # 去除 <s/> 及其后的 token
                    truncated_length = len(truncated_tokens)  # 计算去除后的长度
                    print(f"Truncated length: {truncated_length}")
                else:
                    print("The token '<s/>' was not found.")
                
                ## blacklist common token selection
                # 基础黑名单符号
                # base_blacklist = ['self', ',', '.', ':', '(', ')', '[', ']', '{', '}', ';']

                # # 为每个符号生成对应的Ġ开头的版本
                # blacklist = base_blacklist + ['Ġ' + token for token in base_blacklist]
                # blacklist_indices = []
                # # 遍历code_tokens_str，检查是否包含在黑名单中的token
                # for i, token_str in enumerate(code_tokens_str):
                #     if any(blacklisted_token in token_str for blacklisted_token in blacklist):
                #         # 如果匹配，将该index加入黑名单index列表
                #         blacklist_indices.append(i)

                # # 初始化存储所有匹配index的字典
                # matched_indices = []

                # # 在code_inputs[local_index]中搜索相同的token值
                # for blacklist_index in blacklist_indices:
                #     token_value = code_inputs[local_index][blacklist_index]
                #     for i, val in enumerate(code_inputs[local_index]):
                #         if val == token_value:
                #             matched_indices.append(i)
                # matched_indices = list(set(matched_indices))

                # 输出结果
                # print("Blacklist token indices:", blacklist_indices)
                # print("All matched indices for each token:", matched_indices)

                # temp_tokens = []
                # for idx in blacklist_indices:
                #     if idx < len(code_tokens_str):
                #         # print(f"Index {idx}: {code_tokens_str[idx]}")
                #         temp_tokens.append(code_tokens_str[idx])
                #     # else:
                #     #     print(f"Index {idx} is out of range.")

                # common_mapped_code_tokens.append(temp_tokens)
                # common_mapped_tokens.append(blacklist_indices)

                # # embedding-based common tokens selection
                # code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
                # nl_outputs = model(nl_inputs=nl_inputs)

                # code_word_embeddings = code_outputs[0][local_index]
                # selected_embeddings = code_word_embeddings[token_ids]

                # # 计算相似度矩阵
                # similarity_matrix = compute_similarity_matrix(selected_embeddings)

                # # 设置阈值并过滤
                # threshold = 0.8
                # similar_pairs = filter_similar_tokens(similarity_matrix, threshold)

                # # # 输出结果
                # # for i, j, similarity in similar_pairs:
                # #     print(f"Token {i} and Token {j} have similarity {similarity:.4f}")

                # # 提取所有出现的索引
                # unique_indices = extract_indices(similar_pairs)
                # mapped_tokens_id = [token_ids[idx] for idx in unique_indices if idx < len(token_ids)]
                # # print(token_ids)
                # # print(mapped_tokens_id)
                # common_mapped_tokens.append(mapped_tokens_id)

                # code_tokens = code_inputs[local_index].tolist()
                # code_tokens_str = [tokenizer.convert_ids_to_tokens(id) for id in code_tokens]
                # # 打印每个索引及其对应的 token
                # temp_tokens = []
                # for idx in mapped_tokens_id:
                #     if idx < len(code_tokens_str):
                #         print(f"Index {idx}: {code_tokens_str[idx]}")
                #         temp_tokens.append(code_tokens_str[idx])
                #     else:
                #         print(f"Index {idx} is out of range.")

                # common_mapped_code_tokens.append(temp_tokens)
    # aaa
    # with open('/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1/tokenized_tokens_black.json', 'w') as f:
    #     json.dump(common_mapped_code_tokens, f)

    with open('/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1/tokenized_id_black.json', 'w') as f:
        json.dump(tokenized_id, f)
    model.train()

    tr_num,aln_loss,atten_loss,best_mrr=0,0,0,0

    align_loss_list = []
    noisy_attention_loss_list = []

    accumulation_steps = 32  # Number of batches to accumulate gradients before updating
    optimizer.zero_grad()  # Reset gradients before starting

    for idx in range(args.num_train_epochs):
        epoch_align_losses = []
        epoch_noisy_attention_losses = []
        epoch_common_losses = []
        if idx == 0:
            output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx+1))
            ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')

            model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
            model.to(args.device)
        for step,batch in enumerate(train_dataloader):
            #get inputs
            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size
            for i in range(len(sample_index)):
                if batch_start_idx <= sample_index[i] < batch_end_idx:
                    local_index = sample_index[i] - batch_start_idx
                    code_inputs = batch[0].to(args.device)  
                    attn_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    nl_inputs = batch[3].to(args.device)
                    ori2cur_pos = batch[4].to(args.device)
                    
                    code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
                    nl_outputs = model(nl_inputs=nl_inputs)

                    total_code_tokens = min(ori2cur_pos[local_index].max().item(), 255)

                    # loss, all_loss, code_vec, nl_vec, align_loss, noisy_attention_loss, common_token_loss = model.alignment(code_inputs, code_outputs, nl_outputs, local_index, sample_align[i], labelled_mapped_tokens[i], common_mapped_tokens, args)
                    # loss, all_loss, code_vec, nl_vec, align_loss, noisy_attention_loss, common_token_loss = model.attention_alignment(code_inputs, code_outputs, nl_outputs, local_index, sample_align[i], labelled_mapped_tokens[i], common_mapped_tokens, args)
                    retrieval_loss, code_vec, nl_vec, align_loss, attention_loss = model.new_alignment(code_inputs, code_outputs, nl_outputs, local_index, sample_align[i], total_code_tokens)
                    # print("align_loss",align_loss)
                    # align_loss = align_loss
                    # noisy_attention_loss = noisy_attention_loss / 10
                    # if align_loss > 0 and noisy_attention_loss > 0:
                    #     aa_loss = align_loss + noisy_attention_loss
                    # elif align_loss > 0:
                    #     aa_loss = align_loss
                    #     noisy_attention_loss = 0.0
                    # else:
                    #     aa_loss = noisy_attention_loss
                    #     align_loss = 0.0

                    # if common_token_loss > 0:
                    #     aa_loss = aa_loss + common_token_loss
                    aa_loss = align_loss + attention_loss + retrieval_loss
                    
                    # 确保 align_loss 和 noisy_attention_loss 是张量
                    if isinstance(align_loss, torch.Tensor):
                        align_loss_value = align_loss.item()
                    else:
                        align_loss_value = align_loss
                    
                    if isinstance(attention_loss, torch.Tensor):
                        attention_loss_value = attention_loss.item()
                    else:
                        attention_loss_value = attention_loss

                    # if isinstance(common_token_loss, torch.Tensor):
                    #     common_token_loss_value = common_token_loss.item()
                    # else:
                    #     common_token_loss_value = common_token_loss

                    epoch_align_losses.append(align_loss_value)
                    epoch_noisy_attention_losses.append(attention_loss_value)
                    # epoch_common_losses.append(common_token_loss_value)
                    
                    # # Accumulate loss
                    # aa_loss = aa_loss / accumulation_steps  # Average loss over accumulation steps

                    #backward
                    # align_loss.backward()
                    aa_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    # optimizer.step()
                    # optimizer.zero_grad()
                    # scheduler.step()
                    def safe_item(value):
                        if isinstance(value, torch.Tensor):
                            return value.item()
                        return value
                    # logger.info("epoch {} step {} alignment loss {} attention loss {}  common loss {}".format(idx,step+1,round(safe_item(align_loss),5),round(safe_item(noisy_attention_loss),5),round(safe_item(common_token_loss_value),5)))
                    logger.info("epoch {} step {} alignment loss {} attention loss {} retrieval loss {}".format(idx,step+1,round(safe_item(align_loss),5),round(safe_item(attention_loss),5),round(safe_item(retrieval_loss),5)))
                    # Update model parameters and zero gradients every accumulation_steps
                    # if (step + 1) % accumulation_steps == 0:
                    #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    #     optimizer.step()
                    #     optimizer.zero_grad()
                    #     scheduler.step()
        # aaa
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # Ensure the last batch updates if it's not a multiple of accumulation_steps
        # if (step + 1) % accumulation_steps != 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     scheduler.step()

        # Calculate average losses for the epoch
        # avg_epoch_align_loss = sum(epoch_align_losses) / len(epoch_align_losses)
        # avg_epoch_noisy_attention_loss = sum(epoch_noisy_attention_losses) / len(epoch_noisy_attention_losses)
        # avg_epoch_common_losses = sum(epoch_common_losses) / len(epoch_common_losses)
        # logger.info("Epoch {} - Average alignment loss: {:.5f}, Average attention loss: {:.5f}, Average common loss: {:.5f}".format(idx + 1, avg_epoch_align_loss, avg_epoch_noisy_attention_loss, avg_epoch_common_losses))

        align_loss_list.append(epoch_align_losses)
        # noisy_attention_loss_list.append(epoch_noisy_attention_losses)

        # if idx % 10 == 0:
        #     #evaluate    
        #     results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        #     for key, value in results.items():
        #         logger.info("  %s = %s", key, round(value,4))    
                
        #     #save best model
        #     if results['eval_mrr']>best_mrr:
        #         best_mrr=results['eval_mrr']
        #         logger.info("  "+"*"*20)  
        #         logger.info("  Best mrr:%s",round(best_mrr,4))
        #         logger.info("  "+"*"*20)                          

        #         checkpoint_prefix = 'checkpoint-best-mrr'
        #         output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
        #         if not os.path.exists(output_dir):
        #             os.makedirs(output_dir)                        
        #         model_to_save = model.module if hasattr(model,'module') else model
        #         output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
        #         torch.save(model_to_save.state_dict(), output_dir)
        #         logger.info("Saving model checkpoint to %s", output_dir)

        output_dir = os.path.join("/home/yiming/cophi/training_dynamic/gcb_tokens/Model", 'Epoch_{}'.format(idx + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_output_path = os.path.join(output_dir, 'subject_model_aa.pth')
        # ckpt_output_path = os.path.join(output_dir, 'subject_model_retrieval.pth')
        logger.info("Saving model checkpoint to %s", ckpt_output_path)
        torch.save(model_to_save.state_dict(), ckpt_output_path)

    # # 保存列表到文件
    # with open('/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1/align_loss_list.json', 'w') as f:
    #     json.dump(align_loss_list, f)

    # with open('/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1/noisy_attention_loss_list.json', 'w') as f:
    #     json.dump(noisy_attention_loss_list, f)

    print("Loss values saved.")

def train_alignment_auto_samples(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()

    input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_sorted.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')  # 去除行末的逗号
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])
    
    sample_index = idx_list
    sample_align = match_list

    # model.train()
    model.eval()

    accumulation_steps = 32  # Number of batches to accumulate gradients before updating
    optimizer.zero_grad()  # Reset gradients before starting

    for idx in range(args.num_train_epochs):
        # 初始指针位置为0
        sample_idx_ptr = 0
        total_epoch_align_loss = 0
        total_epoch_attention_loss = 0
        total_epoch_retrieval_loss = 0
        # total_epoch_mlm_loss = 0
        total_batches_with_align_loss = 0
        total_batches = 0
        # if idx == 0:
        #     output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx+1))
        #     ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')
        #     model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
        #     model.to(args.device)
        for step,batch in enumerate(train_dataloader):
            total_batches += 1
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            ori2cur_pos = batch[4].to(args.device)
            
            code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            nl_outputs = model(nl_inputs=nl_inputs)
            
            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size

            bs = code_inputs.shape[0]
            code_vec = code_outputs[1]
            nl_vec = nl_outputs[1]

            scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            retrieval_loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            total_epoch_retrieval_loss += retrieval_loss.item()

            # 初始化 align_loss 和 attention_loss 累加变量
            total_align_loss = 0
            total_attention_loss = 0
            # total_mlm_loss = 0
            align_loss_count = 0  # 记录 align_loss 的数量

            # 从当前指针位置开始检查 sample_index 是否在当前 batch 内
            while sample_idx_ptr < len(sample_index) and sample_index[sample_idx_ptr] < batch_end_idx:
                if sample_index[sample_idx_ptr] >= batch_start_idx:
                    # 计算在当前 batch 范围内的 align_loss 和 attention_loss
                    local_index = sample_index[sample_idx_ptr] - batch_start_idx
                    total_code_tokens = min(ori2cur_pos[local_index].max().item(), 255)

                    if isinstance(model, torch.nn.DataParallel):
                        align_loss, attention_loss = model.module.batch_alignment(
                            code_inputs, code_outputs, nl_outputs, local_index, sample_align[sample_idx_ptr], total_code_tokens
                        )
                    else:
                        align_loss, attention_loss = model.batch_alignment(
                            code_inputs, code_outputs, nl_outputs, local_index, sample_align[sample_idx_ptr], total_code_tokens
                        )
                    
                    # # add mlm loss
                    # if isinstance(model, torch.nn.DataParallel):
                    #     align_loss, attention_loss, mlm_loss = model.module.batch_alignment_mlm(
                    #         code_inputs, code_outputs, nl_outputs, local_index, sample_align[sample_idx_ptr], total_code_tokens, tokenizer, attn_mask[local_index], position_idx[local_index], model
                    #     )
                    # else:
                    #     align_loss, attention_loss, mlm_loss = model.batch_alignment_mlm(
                    #         code_inputs, nl_inputs, code_outputs, nl_outputs, local_index, sample_align[sample_idx_ptr], total_code_tokens, tokenizer, attn_mask[local_index], position_idx[local_index], model
                    #     )

                    total_align_loss += align_loss
                    total_attention_loss += 0.1 * attention_loss
                    # total_mlm_loss += mlm_loss
                    align_loss_count += 1

                sample_idx_ptr += 1  # 移动指针到下一个 sample_index

            # 如果当前 batch 有 align_loss 和 attention_loss，则加上 retrieval_loss 进行反向传播
            if align_loss_count > 0:
                aa_loss = 3*total_align_loss + total_attention_loss + retrieval_loss
                # aa_loss = total_align_loss + total_attention_loss + retrieval_loss + total_mlm_loss
                aa_loss.backward()
                # retrieval_loss.backward()
                total_epoch_align_loss += total_align_loss.item()
                total_epoch_attention_loss += total_attention_loss.item()
                # total_epoch_mlm_loss += total_mlm_loss.item()
                total_batches_with_align_loss += 1
                logger.info(
                    "epoch {} step {} accumulated align loss {} attention loss {} retrieval loss {}".format(
                        idx, step + 1,
                        round(total_align_loss.item(), 5),
                        round(total_attention_loss.item(), 5),
                        round(retrieval_loss.item(), 5)
                        # round(total_mlm_loss.item(), 5)
                    )
                )
            else:
                # 只有 retrieval_loss 的情况
                retrieval_loss.backward()
                logger.info(
                    "epoch {} step {} retrieval loss only {}".format(
                        idx, step + 1,
                        round(retrieval_loss.item(), 5)
                    )
                )
       
            # Update model parameters and zero gradients every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
        # Calculate average losses for the epoch
        avg_align_loss = total_epoch_align_loss / total_batches_with_align_loss if total_batches_with_align_loss > 0 else 0
        avg_attention_loss = total_epoch_attention_loss / total_batches_with_align_loss if total_batches_with_align_loss > 0 else 0
        # avg_mlm_loss = total_epoch_mlm_loss / total_batches_with_align_loss if total_batches_with_align_loss > 0 else 0
        avg_retrieval_loss = total_epoch_retrieval_loss / total_batches

        logger.info("Epoch {} average losses - Align Loss: {:.5f}, Attention Loss: {:.5f}, Retrieval Loss: {:.5f}".format(
            idx + 1, avg_align_loss, avg_attention_loss, avg_retrieval_loss
        ))

        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    

        output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_output_path = os.path.join(output_dir, 'subject_model_new_auto_label_aa_align.pth')
        logger.info("Saving model checkpoint to %s", ckpt_output_path)
        torch.save(model_to_save.state_dict(), ckpt_output_path)

        print("Model saved.")

def train_alignment_auto_labeled_samples(args, model, tokenizer,pool):
    """ Train the model """
    # Load labeled samples first
    input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_sorted_2.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])

    # Get full training dataset
    full_dataset = TextDataset(tokenizer, args, args.train_data_file, pool)
    
    # Extract labeled samples to create new dataset
    labeled_samples = []
    for idx in idx_list:
        labeled_samples.append(full_dataset[idx])
    
    # Create dataloader for labeled samples only
    train_sampler = SequentialSampler(labeled_samples)
    train_dataloader = DataLoader(labeled_samples, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    # Get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load checkpoint if resuming from a specific epoch
    start_epoch = 0
    resume_epoch = None
    if resume_epoch is not None:
        checkpoint_path = os.path.join(args.output_dir, f'Epoch_{resume_epoch}', 'subject_model_only_multiple_aa_labeled.pth')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from epoch {resume_epoch}")
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(torch.load(checkpoint_path), strict=False)
            else:
                model.load_state_dict(torch.load(checkpoint_path), strict=False)
            start_epoch = resume_epoch
        else:
            logger.warning(f"Checkpoint for epoch {resume_epoch} not found. Starting from beginning.")

    # Train!
    logger.info("***** Running training on labeled samples only *****")
    logger.info("  Num labeled examples = %d", len(labeled_samples))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    if resume_epoch:
        logger.info("  Resuming from epoch %d", resume_epoch)
    
    model.zero_grad()
    model.train()
    
    accumulation_steps = 8  # 设置为8,这样每个epoch会更新大约20-25次参数,可以在训练稳定性和效率之间取得平衡
    optimizer.zero_grad()

    # 设定开始加入retrieval loss的轮次
    retrieval_start_epoch = 3

    for idx in range(start_epoch, args.num_train_epochs):
        total_epoch_align_loss = 0
        total_epoch_attention_loss = 0 
        total_epoch_retrieval_loss = 0
        total_batches = 0
        # 当开始retrieval loss后冻结encoder参数并调整学习率
        if idx >= retrieval_start_epoch:
            # 冻结encoder参数并从optimizer中移除
            if isinstance(model, torch.nn.DataParallel):
                for param in model.module.encoder.parameters():
                    param.requires_grad = False
                    # 清除encoder部分的梯度
                    if param.grad is not None:
                        param.grad.zero_()
                        param.grad = None
                # 重建optimizer,只包含未冻结的参数
                optimizer = AdamW([p for n, p in model.module.named_parameters() if 'encoder' not in n and p.requires_grad], 
                                lr=args.learning_rate, eps=1e-8)
            else:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                    # 清除encoder部分的梯度
                    if param.grad is not None:
                        param.grad.zero_()
                        param.grad = None
                # 重建optimizer,只包含未冻结的参数
                optimizer = AdamW([p for n, p in model.named_parameters() if 'encoder' not in n and p.requires_grad],
                                lr=args.learning_rate, eps=1e-8)
            
            # 计算学习率衰减因子
            decay_factor = 0.95 ** (idx - retrieval_start_epoch)
            
            # 调整学习率 - 先增大后逐渐衰减
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * 5.0 * decay_factor  # 初始增大5倍后逐步衰减
        
        for step, batch in enumerate(train_dataloader):
            total_batches += 1
            
            # Get inputs
            code_inputs = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            ori2cur_pos = batch[4].to(args.device)
            
            code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            nl_outputs = model(nl_inputs=nl_inputs)
            
            # 初始化存储所有样本的new cls tokens的tensor
            batch_size = code_inputs.size(0)
            hidden_size = code_outputs[0].size(-1)
            code_cls_new_batch = torch.zeros(batch_size, hidden_size).to(args.device)
            nl_cls_new_batch = torch.zeros(batch_size, hidden_size).to(args.device)

            total_align_loss = 0
            total_attention_loss = 0

            # Process each sample in batch
            for batch_idx in range(code_inputs.size(0)):
                sample_idx = step * args.train_batch_size + batch_idx
                if sample_idx >= len(match_list):
                    break
                    
                total_code_tokens = min(ori2cur_pos[batch_idx].max().item(), 255)
                # Get total tokens for code and comment
                code_tokens_2 = (code_inputs[batch_idx] == 2).nonzero().flatten()
                nl_tokens_2 = (nl_inputs[batch_idx] == 2).nonzero().flatten()
                
                # Set default values if no token 2 found
                if len(code_tokens_2) == 0:
                    total_code_tokens = 255 - 1
                else:
                    total_code_tokens = int(code_tokens_2[0].item() - 1)
                    
                if len(nl_tokens_2) == 0:
                    total_comment_tokens = 127 - 1
                else:
                    total_comment_tokens = int(nl_tokens_2[0].item() - 1)

                if isinstance(model, torch.nn.DataParallel):
                    align_loss, attention_loss, code_cls_new, nl_cls_new = model.module.batch_alignment(
                        code_inputs, code_outputs, nl_outputs, batch_idx, match_list[sample_idx], total_code_tokens, total_comment_tokens
                    )
                else:
                    align_loss, attention_loss, code_cls_new, nl_cls_new = model.batch_alignment(
                        code_inputs, code_outputs, nl_outputs, batch_idx, match_list[sample_idx], total_code_tokens, total_comment_tokens
                    )

                total_align_loss += align_loss
                total_attention_loss += attention_loss

                # 将每个样本的new cls token存入batch tensor中
                code_cls_new_batch[batch_idx] = code_cls_new
                nl_cls_new_batch[batch_idx] = nl_cls_new

            # 计算retrieval loss (在指定轮次后启用)
            if idx >= retrieval_start_epoch:
                scores = torch.einsum("ab,cb->ac",nl_cls_new_batch,code_cls_new_batch)
                temperature = 0.01  # 试试更小的温度
                scores = scores / temperature
                # print("scores min:", scores.min().item(), "scores max:", scores.max().item())
                loss_fct = CrossEntropyLoss()
                retrieval_loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
                total_epoch_retrieval_loss += retrieval_loss.item()
                # 只使用retrieval loss进行反向传播
                retrieval_loss.backward()
            else:
                # 使用原来的loss组合
                aa_loss = 3*total_align_loss + total_attention_loss
                aa_loss.backward()
            
            total_epoch_align_loss += total_align_loss.item()
            total_epoch_attention_loss += total_attention_loss.item()
            
            if idx >= retrieval_start_epoch:
                logger.info(
                    "epoch {} step {} retrieval loss {}".format(
                        idx, step + 1,
                        round(retrieval_loss.item(), 5)
                    )
                )
            else:
                logger.info(
                    "epoch {} step {} accumulated align loss {} attention loss {}".format(
                        idx, step + 1,
                        round(total_align_loss.item(), 5),
                        round(total_attention_loss.item(), 5)
                    )
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Calculate average losses for the epoch
        avg_align_loss = total_epoch_align_loss / total_batches
        avg_attention_loss = total_epoch_attention_loss / total_batches
        avg_retrieval_loss = total_epoch_retrieval_loss / total_batches if idx >= retrieval_start_epoch else 0

        if idx >= retrieval_start_epoch:
            logger.info("Epoch {} average losses - Retrieval Loss: {:.5f}".format(
                idx + 1, avg_retrieval_loss
            ))
        else:
            logger.info("Epoch {} average losses - Align Loss: {:.5f}, Attention Loss: {:.5f}".format(
                idx + 1, avg_align_loss, avg_attention_loss
            ))

        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    

        # Save checkpoint
        output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_output_path = os.path.join(output_dir, 'subject_model_only_multiple_aa_labeled_test.pth')
        logger.info("Saving model checkpoint to %s", ckpt_output_path)
        torch.save(model_to_save.state_dict(), ckpt_output_path)

        print("Model saved.")

def save_labeled_samples_cls_tokens(args, model, tokenizer, pool):
    """ Save new CLS tokens for labeled samples across epochs """
    # Load sample indices from file
    input_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyiming-240108540153/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_sorted_2.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])

    # Get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, pool)
    
    # Multi-gpu setup
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Track features across epochs 1-20
    for epoch in range(1, 21):
        # Load model checkpoint for current epoch
        output_dir = os.path.join(args.output_dir, f'Epoch_{epoch}')
        ckpt_output_path = os.path.join(output_dir, 'subject_model_only_multiple_aa_labeled_test.pth')
        
        # Create output directory for features
        features_output_dir = os.path.join("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyiming-240108540153/training_dynamic/features/only_aa_labeled_multi_epochs_new_cls", f'Epoch_{epoch}')
        os.makedirs(features_output_dir, exist_ok=True)

        # Load model state
        model.load_state_dict(torch.load(ckpt_output_path), strict=False)
        model.to(args.device)
        model.eval()

        # Create subset of labeled samples
        labeled_indices = torch.tensor(idx_list)
        labeled_dataset = torch.utils.data.Subset(train_dataset, labeled_indices)
        labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.train_batch_size,
                                      shuffle=False, num_workers=4)

        nl_cls_tokens = []
        code_cls_tokens = []

        # Process all batches
        with torch.no_grad():
            for step, batch in enumerate(labeled_dataloader):
                code_inputs = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                nl_inputs = batch[3].to(args.device)
                ori2cur_pos = batch[4].to(args.device)

                code_outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
                nl_outputs = model(nl_inputs=nl_inputs)

                # Process each sample in batch
                for batch_idx in range(code_inputs.size(0)):
                    sample_idx = step * args.train_batch_size + batch_idx
                    if sample_idx >= len(match_list):
                        break

                    # Get total tokens
                    code_tokens_2 = (code_inputs[batch_idx] == 2).nonzero().flatten()
                    nl_tokens_2 = (nl_inputs[batch_idx] == 2).nonzero().flatten()

                    total_code_tokens = 255 - 1 if len(code_tokens_2) == 0 else int(code_tokens_2[0].item() - 1)
                    total_comment_tokens = 127 - 1 if len(nl_tokens_2) == 0 else int(nl_tokens_2[0].item() - 1)

                    # Get new CLS tokens
                    if isinstance(model, torch.nn.DataParallel):
                        code_cls_new, nl_cls_new = model.module.get_new_cls_tokens(
                            code_outputs, nl_outputs, batch_idx, 
                            total_code_tokens, total_comment_tokens
                        )
                    else:
                        code_cls_new, nl_cls_new = model.get_new_cls_tokens(
                            code_outputs, nl_outputs, batch_idx, 
                            total_code_tokens, total_comment_tokens
                        )

                    code_cls_tokens.append(code_cls_new.cpu().numpy())
                    nl_cls_tokens.append(nl_cls_new.cpu().numpy())

                if (step + 1) % 100 == 0:
                    logger.info(f"Processed {step + 1} batches")

            # Save concatenated features
            nl_cls_tokens = np.array(nl_cls_tokens)
            code_cls_tokens = np.array(code_cls_tokens)
            print(nl_cls_tokens.shape)
            print(code_cls_tokens.shape)

            np.save(os.path.join(features_output_dir, 'train_nl_cls_tokens.npy'), nl_cls_tokens)
            np.save(os.path.join(features_output_dir, 'train_code_cls_tokens.npy'), code_cls_tokens)

            print(f"Saved features for epoch {epoch}")

        # Clear GPU memory
        torch.cuda.empty_cache()

def save_labeled_sample_cl_valid(args, model, tokenizer, pool):
    """ Save features for all validation samples across all epochs """
    # Get validation datasets
    valid_nl_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool)
    valid_code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
    
    valid_nl_dataloader = DataLoader(valid_nl_dataset, batch_size=args.train_batch_size, 
                                   shuffle=False, num_workers=4)
    valid_code_dataloader = DataLoader(valid_code_dataset, batch_size=args.train_batch_size,
                                     shuffle=False, num_workers=4)

    # Multi-gpu setup
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Track features across epochs
    for epoch in range(5, 6):
        # Load model checkpoint for current epoch
        output_dir = os.path.join(args.output_dir, f'Epoch_{epoch}')
        ckpt_output_path = os.path.join(output_dir, 'subject_model_labeled_retrieval_only_2.pth')
        
        # Create output directory for features
        features_output_dir = os.path.join("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyiming-240108540153/training_dynamic/features/only_aa_labeled_multi_epochs_cls_valid", f'Epoch_{epoch}')
        os.makedirs(features_output_dir, exist_ok=True)
        
        # Load model state
        model.load_state_dict(torch.load(ckpt_output_path), strict=False)
        model.to(args.device)
        model.eval()

        # Process NL features first
        nl_cls_tokens = []
        with torch.no_grad():
            for step, batch in enumerate(valid_nl_dataloader):
                nl_inputs = batch[3].to(args.device)
                nl_outputs = model(nl_inputs=nl_inputs)

                for batch_idx in range(nl_inputs.size(0)):
                    nl_tokens_2 = (nl_inputs[batch_idx] == 2).nonzero().flatten()
                    total_comment_tokens = 127 - 1 if len(nl_tokens_2) == 0 else int(nl_tokens_2[0].item() - 1)

                    if isinstance(model, torch.nn.DataParallel):
                        nl_cls_new = model.module.get_new_cls_tokens_for_code_or_nl(
                            nl_outputs, batch_idx,
                            total_comment_tokens
                        )
                    else:
                        nl_cls_new = model.get_new_cls_tokens_for_code_or_nl(
                            nl_outputs, batch_idx,
                            total_comment_tokens
                        )

                    nl_cls_tokens.append(nl_cls_new.cpu().numpy())

                if (step + 1) % 100 == 0:
                    logger.info(f"Processed {step + 1} NL batches")

            # Save NL features
            nl_cls_tokens = np.array(nl_cls_tokens)
            print(f"NL features shape: {nl_cls_tokens.shape}")
            np.save(os.path.join(features_output_dir, 'valid_nl_cls_tokens.npy'), nl_cls_tokens)
            del nl_cls_tokens
            print(f"Saved NL validation features for epoch {epoch}")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Process code features
        code_cls_tokens = []
        with torch.no_grad():
            for step, batch in enumerate(valid_code_dataloader):
                code_inputs = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                code_outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)

                for batch_idx in range(code_inputs.size(0)):
                    code_tokens_2 = (code_inputs[batch_idx] == 2).nonzero().flatten()
                    total_code_tokens = 255 - 1 if len(code_tokens_2) == 0 else int(code_tokens_2[0].item() - 1)

                    if isinstance(model, torch.nn.DataParallel):
                        code_cls_new = model.module.get_new_cls_tokens_for_code_or_nl(
                            code_outputs, batch_idx,
                            total_code_tokens
                        )
                    else:
                        code_cls_new = model.get_new_cls_tokens_for_code_or_nl(
                            code_outputs, batch_idx,
                            total_code_tokens
                        )

                    code_cls_tokens.append(code_cls_new.cpu().numpy())

                if (step + 1) % 100 == 0:
                    logger.info(f"Processed {step + 1} code batches")

            # Save code features
            code_cls_tokens = np.array(code_cls_tokens)
            print(f"Code features shape: {code_cls_tokens.shape}")
            np.save(os.path.join(features_output_dir, 'valid_code_cls_tokens.npy'), code_cls_tokens)
            del code_cls_tokens
            print(f"Saved code validation features for epoch {epoch}")

        # Clear GPU memory
        torch.cuda.empty_cache()



def train_alignment_single_sample(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()

    input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/sorted_labelling_sample_api.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')  # 去除行末的逗号
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])
    
    sample_index = idx_list
    sample_align = match_list

    model.train()

    accumulation_steps = 32  # Number of batches to accumulate gradients before updating
    optimizer.zero_grad()  # Reset gradients before starting

    for idx in range(args.num_train_epochs):
        # 初始指针位置为0
        sample_idx_ptr = 0
        total_epoch_align_loss = 0
        total_epoch_attention_loss = 0
        total_batches_with_align_loss = 0
        total_batches = 0

        for step,batch in enumerate(train_dataloader):
            total_batches += 1            
            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size

            # 初始化 align_loss 和 attention_loss 累加变量
            total_align_loss = 0
            total_attention_loss = 0
            align_loss_count = 0  # 记录 align_loss 的数量

            # 从当前指针位置开始检查 sample_index 是否在当前 batch 内
            while sample_idx_ptr < len(sample_index) and sample_index[sample_idx_ptr] < batch_end_idx:
                if sample_index[sample_idx_ptr] >= batch_start_idx:

                    #get inputs
                    code_inputs = batch[0].to(args.device)  
                    attn_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    nl_inputs = batch[3].to(args.device)
                    ori2cur_pos = batch[4].to(args.device)
                    
                    code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
                    nl_outputs = model(nl_inputs=nl_inputs)

                    # 计算在当前 batch 范围内的 align_loss 和 attention_loss
                    local_index = sample_index[sample_idx_ptr] - batch_start_idx
                    total_code_tokens = min(ori2cur_pos[local_index].max().item(), 255)

                    if isinstance(model, torch.nn.DataParallel):
                        align_loss, attention_loss = model.module.batch_alignment(
                            code_inputs, code_outputs, nl_outputs, local_index, sample_align[sample_idx_ptr], total_code_tokens
                        )
                    else:
                        align_loss, attention_loss = model.batch_alignment(
                            code_inputs, code_outputs, nl_outputs, local_index, sample_align[sample_idx_ptr], total_code_tokens
                        )

                    total_align_loss += align_loss
                    total_attention_loss += attention_loss
                    align_loss_count += 1

                    # 如果当前 batch 有 align_loss 和 attention_loss，则进行反向传播
                    if align_loss_count > 0:
                        loss = total_align_loss + total_attention_loss
                        loss.backward()
                        total_epoch_align_loss += total_align_loss.item()
                        total_epoch_attention_loss += total_attention_loss.item()
                        total_batches_with_align_loss += 1
                        logger.info(
                            "epoch {} step {} accumulated align loss {} attention loss {}".format(
                                idx, step + 1,
                                round(total_align_loss.item(), 5),
                                round(total_attention_loss.item(), 5)
                            )
                        )
            
                    # Update model parameters and zero gradients every epoch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                sample_idx_ptr += 1
        # Calculate average losses for the epoch
        avg_align_loss = total_epoch_align_loss / total_batches_with_align_loss if total_batches_with_align_loss > 0 else 0
        avg_attention_loss = total_epoch_attention_loss / total_batches_with_align_loss if total_batches_with_align_loss > 0 else 0

        logger.info("Epoch {} average losses - Align Loss: {:.5f}, Attention Loss: {:.5f}".format(
            idx + 1, avg_align_loss, avg_attention_loss
        ))

        output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_output_path = os.path.join(output_dir, 'subject_model_new_auto_label_aa_one_sample.pth')
        logger.info("Saving model checkpoint to %s", ckpt_output_path)
        torch.save(model_to_save.state_dict(), ckpt_output_path)

        print("Model saved.")
    
    # Evaluate
    results = evaluate(args, model, tokenizer, args.eval_data_file, pool, eval_when_training=True)
    for key, value in results.items():
        logger.info("  %s = %s", key, round(value, 4))    

def train_alignment_retrieval_only(args, model, tokenizer, pool):
    """ Train the model with retrieval loss only for matched samples """
    # Get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    # Get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()

    # Load sample indices
    input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/sorted_labelling_sample_api.jsonl"
    idx_list = []
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
    
    sample_index = idx_list
    model.train()

    accumulation_steps = 32
    optimizer.zero_grad()

    for idx in range(args.num_train_epochs):
        sample_idx_ptr = 0
        total_epoch_retrieval_loss = 0
        total_batches_with_retrieval_loss = 0
        total_batches = 0

        for step, batch in enumerate(train_dataloader):
            total_batches += 1
            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size

            # 从当前指针位置开始检查 sample_index 是否在当前 batch 内
            while sample_idx_ptr < len(sample_index) and sample_index[sample_idx_ptr] < batch_end_idx:
                if sample_index[sample_idx_ptr] >= batch_start_idx:
                    # 计算在当前 batch 范围内的 align_loss 和 attention_loss
                    local_index = sample_index[sample_idx_ptr] - batch_start_idx
            
                    code_inputs = batch[0].to(args.device)  
                    attn_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    nl_inputs = batch[3].to(args.device)
                    
                    code_outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
                    nl_outputs = model(nl_inputs=nl_inputs)
                    
                    code_vec = code_outputs[1]
                    nl_vec = nl_outputs[1]

                    scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
                    loss_fct = CrossEntropyLoss()
                    retrieval_loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
                    
                    retrieval_loss.backward()
                    total_epoch_retrieval_loss += retrieval_loss.item()
                    total_batches_with_retrieval_loss += 1

                    logger.info(
                        "epoch {} step {} retrieval loss {}".format(
                            idx, step + 1,
                            round(retrieval_loss.item(), 5)
                        )
                    )

                    # Update parameters every accumulation_steps
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                sample_idx_ptr += 1  # 移动指针到下一个 sample_index
        # Calculate average loss for the epoch
        avg_retrieval_loss = total_epoch_retrieval_loss / total_batches_with_retrieval_loss if total_batches_with_retrieval_loss > 0 else 0
        logger.info("Epoch {} average loss - Retrieval Loss: {:.5f}".format(
            idx + 1, avg_retrieval_loss
        ))

        # Save model
        output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_output_path = os.path.join(output_dir, 'subject_model_retrieval_only.pth')
        logger.info("Saving model checkpoint to %s", ckpt_output_path)
        torch.save(model_to_save.state_dict(), ckpt_output_path)

        print("Model saved.")
    
    # Evaluate
    results = evaluate(args, model, tokenizer, args.eval_data_file, pool, eval_when_training=True)
    for key, value in results.items():
        logger.info("  %s = %s", key, round(value, 4))    


def save_tokenized_strs(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()
    model.eval()
    code_tokens_strs = []
    nl_tokens_strs = []
    for step, batch in enumerate(train_dataloader):
        code_inputs = batch[0].to(args.device)  
        nl_inputs = batch[3].to(args.device)
        # 遍历当前 batch 中每个数据，提取 tokens str
        for local_index in range(len(code_inputs)):  # 假设每个 batch 中有多个数据
            code_tokens = code_inputs[local_index].tolist()
            code_tokens_str = [tokenizer.convert_ids_to_tokens(id) for id in code_tokens]

            nl_tokens = nl_inputs[local_index].tolist()
            nl_tokens_str = [tokenizer.convert_ids_to_tokens(id) for id in nl_tokens]
            
            # 将当前的 tokens str 添加到总列表中
            code_tokens_strs.append(code_tokens_str)
            nl_tokens_strs.append(nl_tokens_str)

    print(len(code_tokens_strs), len(nl_tokens_strs))

    with open('/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_code_tokens_train.json', 'w') as f:
        json.dump(code_tokens_strs, f)
    
    with open('/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_comment_tokens_train.json', 'w') as f:
        json.dump(nl_tokens_strs, f)

def save_multi_valid_comment_tokens_embedding(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.eval_data_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    sample_index = 5783
    nl_tokens_strs = []
    # # for first run, we store all cls tokens and attention features
    for step,batch in enumerate(train_dataloader):
        nl_inputs = batch[3].to(args.device)

        for nl_input in nl_inputs:
            nl_token_ids = nl_input.tolist()
            nl_tokens_str = [tokenizer.convert_ids_to_tokens(token_id) for token_id in nl_token_ids]
            nl_tokens_strs.append(nl_tokens_str)

    print(len(nl_tokens_strs))

    with open('/home/yiming/cophi/training_dynamic/temp_featrues/valid_tokenized_nl_tokens.json', 'w') as f:
        json.dump(nl_tokens_strs, f)

    for iter in range(30):
        output_dir = os.path.join("/home/yiming/cophi/training_dynamic/gcb_tokens/Model", 'Epoch_{}'.format(iter+1))
        ckpt_output_path = os.path.join(output_dir, 'subject_model_new_batch_retrieval.pth')

        model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
        model.to(args.device)
        nl_tokens = []

        for step,batch in enumerate(train_dataloader):
            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size
            if batch_start_idx <= sample_index < batch_end_idx:
                nl_inputs = batch[3].to(args.device)
                nl_outputs = model(nl_inputs=nl_inputs)
                nl_last_hidden_state = nl_outputs[0]
                local_index = sample_index - batch_start_idx
                nl_tokens.append(nl_last_hidden_state[local_index].unsqueeze(0).cpu().detach().numpy())
            
        nl_tokens = np.concatenate(nl_tokens, 0)

        output_dir = os.path.join("/home/yiming/cophi/training_dynamic/temp_featrues", 'Epoch_{}'.format(iter+1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
        
        print(nl_tokens.shape)
        nl_tokens_output_path = os.path.join(output_dir, 'new_valid_nl_tokens_new_batch_retrieval.npy')
        np.save(nl_tokens_output_path, nl_tokens)

def save_multi_valid_code_tokens_embedding(args, model, tokenizer,pool):
    torch.cuda.empty_cache()
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.codebase_file, pool)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()

    
    code_tokens_strs = []
    # set code index here to store the corresponding token feature
    sample_index = 30273

    # # for first run, we store all cls tokens and attention features
    for step,batch in enumerate(train_dataloader):
        code_inputs = batch[0].to(args.device)  

        for code_input in code_inputs:
            # code_input 是一个包含多个 token IDs 的列表
            code_token_ids = code_input.tolist()
            code_tokens_str = [tokenizer.convert_ids_to_tokens(token_id) for token_id in code_token_ids]
            code_tokens_strs.append(code_tokens_str)

    print(len(code_tokens_strs))
    with open('/home/yiming/cophi/training_dynamic/temp_featrues/valid_tokenized_code_tokens.json', 'w') as f:
        json.dump(code_tokens_strs, f)

    # for observation, we store certain code tokens and tokenized ids
    for iter in range(30):
        tokenized_id = []
        code_tokens = []
        output_dir = os.path.join("/home/yiming/cophi/training_dynamic/gcb_tokens/Model", 'Epoch_{}'.format(iter+1))
        ckpt_output_path = os.path.join(output_dir, 'subject_model_new_batch_retrieval.pth')
        model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
        model.to(args.device)

        for step,batch in enumerate(train_dataloader):
            batch_start_idx = step * args.train_batch_size
            batch_end_idx = batch_start_idx + args.train_batch_size
            if batch_start_idx <= sample_index < batch_end_idx:
                code_inputs = batch[0].to(args.device)  
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                code_outputs = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
                ori2cur_pos = batch[4].to(args.device)

                local_index = sample_index - batch_start_idx

                tokenized_id.append(ori2cur_pos[local_index].tolist())
                code_last_hidden_state = code_outputs[0]
                code_tokens.append(code_last_hidden_state[local_index].unsqueeze(0).cpu().detach().numpy())

        code_tokens = np.concatenate(code_tokens, 0)

        output_dir = os.path.join("/home/yiming/cophi/training_dynamic/temp_featrues", 'Epoch_{}'.format(iter+1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  

        print(code_tokens.shape)
        code_token_output_path = os.path.join(output_dir, 'new_valid_code_tokens_new_align_attention_batch_retrieval.npy')
        np.save(code_token_output_path, code_tokens)

        with open('/home/yiming/cophi/training_dynamic/temp_featrues/tokenized_id_valid.json', 'w') as f:
            json.dump(tokenized_id, f)

def extract_indices(similar_pairs):
    """
    从相似度对中提取所有出现的 i 和 j 索引，并去重。
    
    Args:
    - similar_pairs (list of tuples): 相似度对的列表，每个元组为 (i, j, similarity)。
    
    Returns:
    - unique_indices (set): 所有出现的唯一索引集合。
    """
    indices = set()
    
    for i, j, _ in similar_pairs:
        indices.add(i)
        indices.add(j)
    
    return indices

def compute_similarity_matrix(embeddings):
    """
    计算给定嵌入矩阵之间的余弦相似度矩阵。
    
    Args:
    - embeddings (torch.Tensor): 嵌入矩阵，形状为 (n, d)，其中 n 是 token 数量, d 是每个 token 的维度。
    
    Returns:
    - similarity_matrix (torch.Tensor): 相似度矩阵，形状为 (n, n)。
    """
    # 计算嵌入的 L2 范数
    norms = embeddings.norm(dim=1, keepdim=True)
    # 计算余弦相似度矩阵
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    return similarity_matrix

def filter_similar_tokens(similarity_matrix, threshold):
    """
    根据相似度矩阵和阈值过滤出相似度高于阈值的 token 对。
    
    Args:
    - similarity_matrix (torch.Tensor): 相似度矩阵，形状为 (n, n)。
    - threshold (float): 相似度阈值。
    
    Returns:
    - similar_pairs (list of tuples): 过滤后的相似 token 对，每个元组为 (i, j, similarity)。
    """
    similar_pairs = []
    n = similarity_matrix.size(0)
    
    for i in range(n):
        for j in range(i + 1, n):  # 只检查上三角矩阵部分以避免重复
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                similar_pairs.append((i, j, similarity))
                
    return similar_pairs

# def filter_similar_tokens(similarity_matrix, threshold):
#     """
#     根据相似度矩阵和阈值过滤出相似度高于阈值的 token 对。
    
#     Args:
#     - similarity_matrix (torch.Tensor): 相似度矩阵，形状为 (n, n)。
#     - threshold (float): 相似度阈值。
    
#     Returns:
#     - similar_pairs (list of tuples): 过滤后的相似 token 对，每个元组为 (i, j, similarity)。
#     """
#     similar_pairs = []
#     n = similarity_matrix.size(0)
    
#     # 查找所有高于阈值的相似对
#     for i in range(n):
#         for j in range(i + 1, n):  # 只检查上三角矩阵部分以避免重复
#             similarity = similarity_matrix[i, j]
#             if similarity > threshold:
#                 similar_pairs.append((i, j, similarity))
    
#     # 如果找到了高于阈值的相似对，直接返回
#     if similar_pairs:
#         return similar_pairs
    
#     # 否则，找出相似度最高的对（排除对角线上的对）
#     # 计算矩阵中非对角线部分的最大相似度
#     mask = torch.eye(n, dtype=torch.bool, device=similarity_matrix.device)
#     masked_similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
#     max_similarity = torch.max(masked_similarity_matrix)
    
#     max_indices = torch.nonzero(masked_similarity_matrix == max_similarity, as_tuple=False)
    
#     # 如果有多个最大值对，选择第一个
#     i, j = max_indices[0]
#     return [(i.item(), j.item(), max_similarity.item())]

def find_tokens_with_first_occurrence(mapped_tokens):
    # 存储每个 token 出现的行数和它们的第一次出现位置
    token_info = defaultdict(lambda: {'count': 0, 'first_occurrence': {}})
    
    # 遍历每一行的 tokens
    for row_index, row in enumerate(mapped_tokens):
        if isinstance(row, torch.Tensor):
            row = row.tolist()
        
        # 记录每个 token 在当前行中的第一次出现位置
        seen_tokens = set()
        for index, token in enumerate(row):
            token = int(token)
            if token not in seen_tokens:
                seen_tokens.add(token)
                if token not in token_info:
                    token_info[token] = {'count': 0, 'first_occurrence': {}}
                token_info[token]['count'] += 1
                token_info[token]['first_occurrence'][row_index] = index
    
    # 筛选出在多行中出现的 token，并记录它们的第一次出现位置
    common_tokens_info = {token: info for token, info in token_info.items() if info['count'] > 1}
    
    return common_tokens_info

def extract_tokens_from_intervals(code_ids, code_tokens_list, ori2cur_pos):
    # 生成一个用于存储结果的列表
    extracted_tokens = []
    extracted_ids = []

    # 遍历 code_tokens_list 中的每个索引 i
    for i in range(len(code_tokens_list)):
        token_temp = []
        id_temp = []
        for j in range(len(code_tokens_list[i])):
            # 获取对应的区间
            start, end = ori2cur_pos[code_tokens_list[i][j]].tolist()
            # print(start, end)
            # 提取该区间内的值
            for k in range(start+1, end+1):
                # print(k)
                tokens_in_range = code_ids[k]
                token_temp.append(tokens_in_range)
                id_temp.append(k)
        
        # 将提取的 tokens 添加到结果列表中
        extracted_tokens.append(token_temp)
        extracted_ids.append(id_temp)

    return extracted_tokens, extracted_ids

def extract_code_tokens_from_alignment(alignment):
    code_tokens_list = []

    for interval in alignment:
        interval_code = interval[1]
        interval_tokens = []
        # 确保每个 interval 是一对 start, end
        for i in range(0, len(interval_code), 2):
            start = interval_code[i]
            end = interval_code[i+1]
            # 提取区间的 tokens
            interval_tokens.extend(range(start+1, end+2))
        code_tokens_list.append(interval_tokens)

    return code_tokens_list

def save_feature(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
        
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            # 每10个step保存一次模型
            if step % 10 == 9:  # 0-based index, so 9, 19, 29, ...
                output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(step + 1))
                ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')
                model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
                model.to(args.device)
                model.eval()

                all_code_outputs = []
                all_nl_outputs = []

                for input_step, batch in enumerate(train_dataloader):
                    if input_step >= 10:
                        break
                    #get inputs
                    code_inputs = batch[0].to(args.device)  
                    attn_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    nl_inputs = batch[3].to(args.device)

                    code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)[1]
                    nl_vec = model(nl_inputs=nl_inputs)[1]
                    all_code_outputs.append(code_vec.cpu().detach().numpy())
                    all_nl_outputs.append(nl_vec.cpu().detach().numpy())

                # 连接前10个batch的输出
                all_code_outputs_np = np.concatenate(all_code_outputs, axis=0)
                all_nl_outputs_np = np.concatenate(all_nl_outputs, axis=0)

                # 将code_outputs和nl_outputs连接起来
                combined_outputs = np.concatenate((all_code_outputs_np, all_nl_outputs_np), axis=0)
                print(combined_outputs.shape)
                # 保存连接后的输出到.npy文件中
                np.save(os.path.join(output_dir, 'train_data.npy'), combined_outputs)
                print(f"Data has been saved to: {output_dir}")
                np.save(os.path.join(output_dir, 'test_data.npy'), combined_outputs) 

def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[3].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)[1] 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        attn_mask = batch[1].to(args.device)
        position_idx =batch[2].to(args.device)
        with torch.no_grad():
            code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)[1]
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks))
    }
    rank_path = os.path.join("/home/yiming/cophi/training_dynamic/graphcodebert/Epoch_1", 'new_valid_rank.json')
    with open(rank_path, 'w') as file:
        json.dump(ranks, file)

    print(f"Data has been saved to: {rank_path}")

    return result

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    # config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)    
    model=Model(model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    # Training
    if args.do_train:
        # train(args, model, tokenizer, pool)
        # train_alignment(args, model, tokenizer, pool)
        # save_feature(args, model, tokenizer, pool)
        # train_alignment_auto_samples(args, model, tokenizer, pool)
        # train_alignment_sample(args, model, tokenizer, pool)
        # save_attention_features(args, model, tokenizer, pool)
        # save_valid_attention_features(args, model, tokenizer, pool)
        # save_valid_attention_features_nl(args, model, tokenizer, pool)
        # save_valid_attention_features_nl_aa(args, model, tokenizer, pool)
        # save_valid_code_attention_features(args, model, tokenizer, pool)
        # save_valid_code_attention_features_aa(args, model, tokenizer, pool)
        # save_attention_features_code(args, model, tokenizer, pool)
        # train_data_preprocess(args, model, tokenizer, pool)
        # save_tokens_features(args, model, tokenizer, pool)
        # save_tokenized_strs(args, model, tokenizer, pool)
        # save_multi_valid_code_tokens_embedding(args, model, tokenizer, pool)
        # save_multi_valid_comment_tokens_embedding(args, model, tokenizer, pool)
        # train_alignment_single_sample(args, model, tokenizer, pool)
        # train_alignment_retrieval_only(args, model, tokenizer, pool)
        # save_train_features(args, model, tokenizer, pool)
        train_alignment_auto_labeled_samples(args, model, tokenizer, pool)
        
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()


