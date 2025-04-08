import torch
import torch.nn.functional as F
import logging
import os
import pickle
import random
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
import argparse
import time
logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing
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

@torch.inference_mode()
def compute_loss(embedding1, embedding2):
    '''
    Compute cosine similarity loss between two embeddings
    Args:
        embedding1: First token embedding tensor (1, hidden_dim)
        embedding2: Second token embedding tensor (1, hidden_dim)
    Returns:
        Cosine similarity loss between embeddings (scalar)
    '''
    # Normalize embeddings 
    embedding1 = F.normalize(embedding1, p=2, dim=1)  # (1, hidden_dim)
    embedding2 = F.normalize(embedding2, p=2, dim=1)  # (1, hidden_dim)
    
    # Compute cosine similarity
    cos_sim = torch.sum(embedding1 * embedding2, dim=1)  # (1,)
    
    # Convert to loss (1 - cos_sim)
    loss = 1 - cos_sim.item()  # scalar
    
    return loss


def token_gradient(
    model,
    query_embed,
    code_embed,
):
    '''
    Compute gradients of similarity between two embeddings with respect to model parameters
    Args:
        model: The model
        query_embed: Query token embedding (B, 1, C)
        code_embed: Reference token embedding (B, 1, C)
    Returns:
        Gradients of classifier layer parameters
    '''
    model.train()
    model.zero_grad()

    # Normalize embeddings
    query_embed = F.normalize(query_embed, dim=-1)
    code_embed = F.normalize(code_embed, dim=-1)

    # Compute cosine similarity
    sim = F.cosine_similarity(query_embed, code_embed, dim=-1).mean()  # Scalar
    
    # Convert to loss (1 - cos_sim) like in compute_loss()
    loss = 1 - sim
    print(loss)
    loss.backward()

    # Get gradients of all model parameters
    model_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            model_grads[name] = param.grad.data.cpu()
            
    # Normalize gradients
    for name in model_grads:
        if len(model_grads[name].shape) >= 2:  # Only normalize matrices
            model_grads[name] = F.normalize(model_grads[name], dim=-1)

    sim.detach()
    model.zero_grad()
    return model_grads

import copy
def slicing(args, model, tokenizer, pool):
    '''
    Load labeled training data and compute loss differences
    '''
    # Load labeled samples first
    input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_sorted.jsonl"
    idx_list = []
    match_list = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().rstrip(',')
            json_obj = json.loads(line)
            idx_list.append(json_obj['idx'])
            match_list.append(json_obj['match'])

    with open("/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/dataset/python/train.jsonl", "r") as f:
        train_dataset = [json.loads(line) for line in f.readlines()]

    # Get full training dataset
    full_dataset = TextDataset(tokenizer, args, args.train_data_file, pool)
    
    # Extract labeled samples to create new dataset
    labeled_samples = []
    for idx in idx_list:
        labeled_samples.append(full_dataset[idx])
    
    # Create dataloader for labeled samples only
    train_sampler = SequentialSampler(labeled_samples)
    train_dataloader = DataLoader(labeled_samples, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    
    full_sampler = SequentialSampler(full_dataset)
    full_dataloader = DataLoader(full_dataset, sampler=full_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    # First pass: Extract embeddings for specific indices
    target_idx = 11637  # Example target sample index
    specific_nl_idx = 8  # Specific token index for NL
    specific_code_idx = 8  # Specific token index for code
    target_found = False

    ckpt_output_path = os.path.join(args.output_dir, 'Epoch_20', 'subject_model_only_multiple_aa_labeled_simple_10k_2.pth')
    model.load_state_dict(torch.load(ckpt_output_path), strict=False)
    model.to(args.device)
    model.eval()

    # Extract embeddings for both code and NL from the same batch
    for batch_idx, batch in enumerate(full_dataloader):
        if batch_idx * args.train_batch_size <= target_idx < (batch_idx + 1) * args.train_batch_size:
            local_idx = target_idx - batch_idx * args.train_batch_size
            
            # Get NL embeddings
            nl_inputs = batch[3].to(args.device)
            nl_outputs = model(nl_inputs=nl_inputs)
            
            # Extract and print NL embedding for target index
            nl_tokens = nl_outputs[2][-2][local_idx]
            print(nl_tokens.shape)
            nl_token_output = nl_tokens[specific_nl_idx]
            print(nl_token_output.shape)

            nl_tokens = tokenizer.convert_ids_to_tokens(nl_inputs[local_idx].cpu().tolist()) 
            print(f"NL token: {nl_tokens[specific_nl_idx]}")

            # Get code embeddings from the same batch
            code_inputs = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            code_outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            
            # Extract and print code embedding for target index
            code_tokens = code_outputs[2][-2][local_idx]
            print(code_tokens.shape)
            code_token_output = code_tokens[specific_code_idx]
            print(code_token_output.shape)

            code_tokens = tokenizer.convert_ids_to_tokens(code_inputs[local_idx].cpu().tolist()) 
            print(f"Code token: {code_tokens[specific_code_idx]}")
            
            target_found = True
            break

    # Record start time
    start_time = time.time()
    
    # Get original model parameters
    theta_orig = {}
    for name, param in model.named_parameters():
        theta_orig[name] = param.data.clone()

    # Compute gradients for target tokens
    target_proj_grad = token_gradient(
        model,
        nl_token_output,
        code_token_output
    )

    model_new = copy.deepcopy(model)
    for name, param in model_new.named_parameters():
        # Only update parameters in layer 10
        if name in target_proj_grad and param.requires_grad:
            param.data = theta_orig[name] + args.learning_rate * target_proj_grad[name].to(param.device)

    # Record end time and print duration
    end_time = time.time()
    print(f"Time taken for parameter updates: {end_time - start_time:.2f} seconds")
            
    if not target_found:
        logger.warning("Target sample not found")
        return None
        
    # Second pass: Compute similarity-based loss changes between original and updated model
    pair_loss_changes = []
    
    # # Store original model state
    # model_orig_state = copy.deepcopy(model.state_dict())
    
    # # Load updated model state  
    # model.load_state_dict(model_new.state_dict())
    model.eval()
    model_new.eval()
    
    for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Processing batches"):
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)
        ori2cur_pos = batch[4].to(args.device)
        batch_size = code_inputs.size(0)
        
        # Get embeddings from both models
        with torch.no_grad():
            # Original model embeddings
            code_outputs_orig = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            nl_outputs_orig = model(nl_inputs=nl_inputs)
            
            # Updated model embeddings
            code_outputs_new = model_new(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            nl_outputs_new = model_new(nl_inputs=nl_inputs)
        
        # Calculate loss changes for each pair in match_list
        for local_idx in range(batch_size):
            # Calculate global sample index
            global_sample_idx = batch_idx * batch_size + local_idx
            
            # Get tokens for current sample
            nl_tokens = tokenizer.convert_ids_to_tokens(nl_inputs[local_idx].cpu().tolist())
            code_tokens = tokenizer.convert_ids_to_tokens(code_inputs[local_idx].cpu().tolist())

            total_code_tokens = min(ori2cur_pos[local_idx].max().item(), 255)
            
            # Get pairs for current sample if they exist
            if global_sample_idx < len(match_list):
                pairs_list = match_list[global_sample_idx]
                
                # Calculate alignment losses and get max alignment pair
                align_loss_orig, max_align_pair_orig = model.batch_alignment_with_max_pair(
                    code_inputs=code_inputs,
                    code_outputs=code_outputs_orig,
                    nl_outputs=nl_outputs_orig,
                    local_index=local_idx,
                    sample_align=pairs_list,
                    total_code_tokens=total_code_tokens
                )
                
                align_loss_new, max_align_pair_new = model_new.batch_alignment_with_max_pair(
                    code_inputs=code_inputs,
                    code_outputs=code_outputs_new,
                    nl_outputs=nl_outputs_new,
                    local_index=local_idx,
                    sample_align=pairs_list,
                    total_code_tokens=total_code_tokens
                )
                
                # Calculate loss change
                loss_change = align_loss_new.item() - align_loss_orig.item()
                
                # Get tokens for max alignment pairs
                nl_token_orig = nl_tokens[max_align_pair_orig[0]]
                code_token_orig = code_tokens[max_align_pair_orig[1]]
                nl_token_new = nl_tokens[max_align_pair_new[0]]
                code_token_new = code_tokens[max_align_pair_new[1]]
                
                pair_loss_changes.append({
                    'sample_idx': global_sample_idx,
                    'loss_change': loss_change,
                    'max_align_orig': (nl_token_orig, code_token_orig),
                    'max_align_new': (nl_token_new, code_token_new)
                })

    # Sort pairs by loss change to get both most negative and most positive values
    pair_loss_changes.sort(key=lambda x: x['loss_change'])
    
    # Print information for samples with the largest loss decreases
    print("\nTop 3 samples with largest loss decreases:")
    for i, change in enumerate(pair_loss_changes[:3]):
        print(f"\nSample {i+1}:")
        print(f"Sample index: {change['sample_idx']}")
        print(f"Loss change: {change['loss_change']:.4f}")
        print(f"Original max alignment: NL token '{change['max_align_orig'][0]}' -> Code token '{change['max_align_orig'][1]}'")
        print(f"New max alignment: NL token '{change['max_align_new'][0]}' -> Code token '{change['max_align_new'][1]}'")
        print(f"Sample code: {train_dataset[idx_list[change['sample_idx']]]['code']}")
    
    # Print information for samples with the largest loss increases
    print("\nTop 3 samples with largest loss increases:")
    for i, change in enumerate(pair_loss_changes[-3:]):
        print(f"\nSample {len(pair_loss_changes)-3+i+1}:")
        print(f"Sample index: {change['sample_idx']}")
        print(f"Loss change: {change['loss_change']:.4f}")
        print(f"Original max alignment: NL token '{change['max_align_orig'][0]}' -> Code token '{change['max_align_orig'][1]}'")
        print(f"New max alignment: NL token '{change['max_align_new'][0]}' -> Code token '{change['max_align_new'][1]}'")
        print(f"Sample code: {train_dataset[idx_list[change['sample_idx']]]['code']}")
    
    negative_loss_count = sum(1 for change in pair_loss_changes if change['loss_change'] < 0)
    positive_loss_count = sum(1 for change in pair_loss_changes if change['loss_change'] > 0)
    print(f"\nSummary Statistics:")
    print(f"Total pairs: {len(pair_loss_changes)}")
    print(f"Negative loss pairs: {negative_loss_count}")
    print(f"Positive loss pairs: {positive_loss_count}")
    # Calculate and print the average loss change
    if pair_loss_changes:
        avg_loss_change = sum(change['loss_change'] for change in pair_loss_changes) / len(pair_loss_changes)
        print(f"Average loss change: {avg_loss_change:.4f}")
    else:
        print("No loss changes to calculate average.")
    return {
        'target_loss_change': pair_loss_changes[0]['loss_change'] if pair_loss_changes else None,
        'all_pair_loss_changes': pair_loss_changes
    }

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file", default=None, type=str, required=True,
                        help="The input evaluation data file (a json file).")
    parser.add_argument("--codebase_file", default=None, type=str, required=True,
                        help="The input codebase data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
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
    cpu_cont = 16
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
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)    
    model=Model(model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    slicing(args, model, tokenizer, pool)

if __name__ == "__main__":
    main()