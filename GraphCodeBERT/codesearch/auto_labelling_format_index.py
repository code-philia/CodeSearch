import os
import numpy as np
import re
from openai import OpenAI
import json

import torch
import matplotlib.pyplot as plt
import ast

from pydantic import BaseModel
from typing import List

class TokenAlignment(BaseModel):
    comment_token: List[str]
    code_token: List[str]

class AlignmentOutput(BaseModel):
    alignments: List[TokenAlignment]

def initialize_centroids(X, k):
    # 随机选择k个初始聚类中心
    indices = torch.randperm(X.size(0))[:k]
    return X[indices]

def compute_distances(X, centroids):
    # 计算每个点到每个聚类中心的欧氏距离
    distances = torch.cdist(X, centroids, p=2)  # 使用L2范数
    return distances

def kmeans(X, k, num_iters=100):
    # 初始化聚类中心
    centroids = initialize_centroids(X, k)
    
    for _ in range(num_iters):
        # 计算距离并为每个点分配最近的聚类中心
        distances = compute_distances(X, centroids)
        labels = distances.argmin(dim=1)
        
        # 重新计算聚类中心
        new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(k)])
        
        # 检查是否有空聚类
        nan_mask = torch.isnan(new_centroids)
        new_centroids[nan_mask] = centroids[nan_mask]  # 如果出现空聚类，保持旧的中心

        # 更新聚类中心
        centroids = new_centroids
    
    # 返回最终的聚类中心和每个点的标签
    return centroids, labels

# 查找顺序匹配的索引 (允许非连续但必须顺序)
def find_ordered_token_indices(tokens, full_token_list):
    """找到匹配的 token 在 full_token_list 中的索引，要求匹配的 token 顺序出现"""
    token_indices = []
    current_index = -1  # 记录当前匹配的位置，初始为 -1
    
    for token in tokens:
        for idx in range(current_index + 1, len(full_token_list)):
            # print(idx, current_index, full_token_list[idx], token)
            if token == full_token_list[idx]:
                token_indices.append(idx)
                current_index = idx  # 更新当前匹配的索引位置
                break
    if len(token_indices) == 0:
        return []
    
    return token_indices

# 将索引序列转换为区间格式
def convert_to_intervals(indices):
    """将索引列表转换为区间格式"""
    if not indices:
        return []
    
    intervals = []
    start = indices[0]
    end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            intervals.append(start)
            intervals.append(end)
            start = indices[i]
            end = indices[i]
    
    intervals.append(start)
    intervals.append(end)
    return intervals



# set openai environ and key
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = "sk-kXKBrhvV20vfBpAdh7cdV8QRIgeS0hXceIuopLc5yEyeERKX"
os.environ["OPENAI_BASE_URL"] = "https://api.key77qiqi.cn/v1"

# load all training data
# 文件路径
file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/dataset/python/train.jsonl'

# 存储所有数据的列表
train_data = []

# 读取 JSONL 文件中的所有数据
with open(file_path, 'r') as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

# load all training data tokens
# 文件路径
file_path = '/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_code_tokens_train.json'

# 读取 JSON 文件
with open(file_path, 'r') as f:
    code_tokens_strs = json.load(f)

# 文件路径
nl_file_path = '/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_comment_tokens_train.json'

# 读取 JSON 文件
with open(nl_file_path, 'r') as f:
    nl_tokens_strs = json.load(f)

# 现在 code_tokens_strs 变量中包含了从 JSON 文件读取的数据
print("len(code_tokens_strs)", len(code_tokens_strs))  # 可以查看加载的数据
print("len(nl_tokens_strs)", len(nl_tokens_strs))  # 可以查看加载的数据

# load training data embeddings (pretrained model)
# Load the embeddings from the stored numpy file
code_token_output_path = "/home/yiming/cophi/training_dynamic/gcb_tokens_temp/train_code_cls_token_pt.npy"
all_embeddings = np.load(code_token_output_path)

print("all_embeddings.shape", all_embeddings.shape) 

# load selected unlabeled indices
random_indices = np.load('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/random_indices.npy')

# load human labeled info
input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/label_human_teacher.jsonl"
idx_list = []
match_list = []

with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().rstrip(',')  # 去除行末的逗号
        json_obj = json.loads(line)
        idx_list.append(json_obj['idx'])
        match_list.append(json_obj['match'])

print("len(idx_list)", len(idx_list)) 

# load already auto labeled info
input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/label_human_auto.jsonl"
auto_idx_list = []

with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().rstrip(',')  # 去除行末的逗号
        json_obj = json.loads(line)
        auto_idx_list.append(json_obj['idx'])

unlabeled_indices = list(set(range(len(all_embeddings))) - set(auto_idx_list)) 
print("len(unlabeled_indices)", len(unlabeled_indices)) 

# # Load already auto labeled indices from label_human_auto_index.jsonl
# unlabeled_indice_raw = []
# with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/label_human_auto_index.jsonl', 'r') as file:
#     for line in file:
#         json_obj = json.loads(line)
#         unlabeled_indice_raw.append(json_obj['idx'])

# unlabeled_indices = list(set(unlabeled_indice_raw))
# # Use the indices from label_human_auto_index.jsonl as unlabeled_indices
# print("len(unlabeled_indices)", len(unlabeled_indices))


# 提取聚类中心的嵌入
cluster_centers = all_embeddings[idx_list]

# 提取未标注样本的嵌入
unlabeled_embeddings = all_embeddings[unlabeled_indices]
# 随机打乱未标注索引的顺序
import random
random.shuffle(unlabeled_indices)

# 初始化用于存储每个未标注样本的最小距离和最近的聚类中心索引
min_distances = []
closest_centers = []

# 逐个计算未标注样本到所有聚类中心的距离，保留最小值
for unlabel_i in unlabeled_indices:
    # 计算当前未标注样本到所有聚类中心的距离
    distances_to_centers = np.linalg.norm(all_embeddings[unlabel_i] - cluster_centers, axis=1)
    # 找到最小距离和对应的聚类中心索引
    min_distance = distances_to_centers.min()
    closest_center = distances_to_centers.argmin()
    
    # 保存最小距离和对应聚类中心
    min_distances.append(min_distance)
    closest_centers.append(closest_center)

# 定义距离阈值
distance_threshold = 2.7  # 这个值可以根据具体需求进行调整

# 筛选出距离小于阈值的样本索引
auto_label_indices = [unlabeled_indices[i] for i in range(len(unlabeled_indices)) if min_distances[i] < distance_threshold]
closest_teachers = [closest_centers[i] for i in range(len(unlabeled_indices)) if min_distances[i] < distance_threshold]
cannotlabeled_indices = list(set(unlabeled_indices) - set(auto_label_indices))

print("len(auto_label_indices)", len(auto_label_indices)) 
print("len(cannotlabeled_indices)", len(cannotlabeled_indices)) 


system_prompt = "You are an expert at aligning tokens between comments and code. You can accurately identify the similarities and differences between tokens, and you are highly skilled at matching tokens based on their semantics and functionality. You are given input data consisting of comment tokens and code tokens, and your task is to align them by identifying concepts in the comments and matching them to corresponding code tokens. Use the example cases below and output your results in the specified format."

# auto labelling 
# for i in range(len(auto_label_indices)):
for auto_label_ind in range(10):
    # construct teacher output
    teacher_ind = closest_teachers[auto_label_ind]
    cur_match_list = match_list[teacher_ind]
    cur_idx = idx_list[teacher_ind]
    
    # Convert tokens to dictionaries with indices
    teach_code_tokens_raw = code_tokens_strs[cur_idx][1:]
    teach_comment_tokens_raw = nl_tokens_strs[cur_idx][1:]
    # Get valid length before </s> token and apply max length limits for teacher tokens
    teach_code_valid_len = min(255, next((i for i, token in enumerate(teach_code_tokens_raw) if token == "</s>"), len(teach_code_tokens_raw)))
    teach_comment_valid_len = min(127, next((i for i, token in enumerate(teach_comment_tokens_raw) if token == "</s>"), len(teach_comment_tokens_raw)))
    
    # Truncate teacher tokens to valid lengths
    teach_code_tokens_raw = teach_code_tokens_raw[:teach_code_valid_len]
    teach_comment_tokens_raw = teach_comment_tokens_raw[:teach_comment_valid_len]
    
    teach_code_tokens_with_index = [f"c_{i}:{token}" for i, token in enumerate(teach_code_tokens_raw)]
    teach_comment_tokens_with_index = [f"d_{i}:{token}" for i, token in enumerate(teach_comment_tokens_raw)]

    teacher_output = ""
    match_idx = 0
    # 遍历 match_list
    for match_item in cur_match_list:
        match_idx += 1
        comment_match = match_item[0]
        code_match = match_item[1]
        
        # Get matched tokens with indices
        matched_comment_tokens = []
        for i in range(0, len(comment_match), 2):
            comment_start, comment_end = comment_match[i], comment_match[i+1]
            matched_comment_tokens.extend(teach_comment_tokens_with_index[comment_start : comment_end + 1])
        
        matched_code_tokens = []
        for i in range(0, len(code_match), 2):
            code_start, code_end = code_match[i], code_match[i+1]
            # Add bounds checking before updating the dictionary
            code_start = min(code_start, teach_code_valid_len - 1)
            code_end = min(code_end, teach_code_valid_len - 1)
            if code_start <= code_end:  # Only add if range is valid
                matched_code_tokens.extend(teach_code_tokens_with_index[code_start : code_end + 1]) 
        
        teacher_output += f"{match_idx}. {matched_comment_tokens}, {matched_code_tokens}\n"

    # construct teacher_prompt
    teacher_prompt = f"""
    Below is an example that demonstrates how to align comment tokens and code tokens:
    **Teacher Example:**
    Comment Tokens Index and Comment Tokens String:
    {teach_comment_tokens_with_index}
    Code Tokens Index and Code Tokens String:
    {teach_code_tokens_with_index}
    **Matching Output:**
    {teacher_output}
    """
    # Log teacher example details to file
    with open("/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/labelling.log", "a") as f:
        f.write("\nTeacher Example Details:\n")
        f.write(f"Teacher index: {cur_idx}\n")
        f.write(f"Comment Tokens: {teach_comment_tokens_with_index}\n")
        f.write(f"Code Tokens: {teach_code_tokens_with_index}\n")
        f.write("Matching Output:\n")
        f.write(teacher_output)
        f.write("----------------------------------------\n")

    # construct student input with indices
    student_idx = auto_label_indices[auto_label_ind]
    student_code_tokens_raw = code_tokens_strs[student_idx][1:]
    student_comment_tokens_raw = nl_tokens_strs[student_idx][1:]

    # Get valid length before </s> token and apply max length limits
    code_valid_len = min(255, next((i for i, token in enumerate(student_code_tokens_raw) if token == "</s>"), len(student_code_tokens_raw)))
    comment_valid_len = min(127, next((i for i, token in enumerate(student_comment_tokens_raw) if token == "</s>"), len(student_comment_tokens_raw)))
    
    # Truncate tokens to valid lengths
    student_code_tokens_raw = student_code_tokens_raw[:code_valid_len]
    student_comment_tokens_raw = student_comment_tokens_raw[:comment_valid_len]
    
    student_code_tokens_with_index = [f"c_{i}:{token}" for i, token in enumerate(student_code_tokens_raw)]
    student_comment_tokens_with_index = [f"d_{i}:{token}" for i, token in enumerate(student_comment_tokens_raw)]

    # construct student_prompt
    student_tokens_part = f"""
    Here are the tokens you need to process:

    Comment Tokens Index and Comment Tokens String:
    {student_comment_tokens_with_index}
    Note: Each comment token is prefixed with "d_i" where i is the index, indicating it comes from the description.

    Code Tokens Index and Code Tokens String:
    {student_code_tokens_with_index}
    Note: Each code token is prefixed with "c_i" where i is the index, indicating it comes from the source code.
    """

    alignment_format = """
    {
        "alignments": [
            {"comment_token": [{"index": "d_i", "token": "token1"}], "code_token": [{"index": "c_i", "token": "tokenA"}]},
            {"comment_token": [{"index": "d_j", "token": "token2"}], "code_token": [{"index": "c_j", "token": "tokenB"}]},
            {"comment_token": [{"index": "d_k", "token": "token3"}], "code_token": [{"index": "c_k", "token": "tokenC"}]}
        ]
    }
    """

    student_prompt = f"""
    CRITICAL ALIGNMENT INSTRUCTIONS:
    1. FIRST analyze comment semantics:
       - Break into distinct concepts
       - Keep concepts separate
       - No combining unrelated ideas
    2. THEN find ALL functional matches:
       - API calls & methods
       - Parameters & returns 
       - Library features
       - Variables & data structures
       - Control flow
       - Only then consider naming
    3. Get complete functional units
    4. One concept per code token
    5. Use exact d_i/c_i indices
    6. Focus on implementation details
    7. MAXIMIZE code token coverage:
       - Try to match every code token possible
       - Only leave tokens unmatched if no semantic connection exists
       - Check all code tokens multiple times to ensure maximum matches

    Here are the tokens to align:
    {student_tokens_part}

    Based on the above instructions and following the teacher example, provide comprehensive alignments between comment concepts and code implementations. Output in this format:
    {alignment_format}
    """

    promt_str = system_prompt + teacher_prompt + student_prompt

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))

    # 打开日志文件
    with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/labelling.log', 'a') as log_file:
        log_file.write(f"\n=== Processing auto_label_ind: {auto_label_ind} ===\n")
        log_file.write(f"Student idx: {student_idx}\n")
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06", 
            messages=[{"role": "user", "content": promt_str}], 
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "alignment_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "alignments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "comment_token": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "index": {"type": "string"},
                                                    "token": {"type": "string"}
                                                },
                                                "required": ["index", "token"],
                                                "additionalProperties": False
                                            }
                                        },
                                        "code_token": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "index": {"type": "string"},
                                                    "token": {"type": "string"}
                                                },
                                                "required": ["index", "token"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["comment_token", "code_token"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["alignments"],
                        "additionalProperties": False
                    }
                }
            },
            max_tokens=500)

        # 记录响应内容
        response_content = response.choices[0].message.content
        log_file.write(f"Response content:\n{response_content}\n")

        # 解析JSON响应
        json_content = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_content:
            json_str = json_content.group(0)
            try:
                alignment_output = json.loads(json_str)
                log_file.write("Successfully parsed JSON response\n")
            except json.JSONDecodeError as e:
                log_file.write(f"JSON parsing error: {e}\n")
                continue
        else:
            log_file.write("No valid JSON content found\n")
            continue

        final_results = []
        log_file.write(f"Teacher index: {teacher_ind}, Student index: {student_idx}\n")

        for alignment in alignment_output["alignments"]:
            comment_indices = [item["index"].replace("d_","") for item in alignment["comment_token"]]
            code_indices = [item["index"].replace("c_","") for item in alignment["code_token"]]
            
            # log_file.write(f"Comment Indices: {comment_indices}\n")
            # log_file.write(f"Code Indices: {code_indices}\n")
            # log_file.write("-" * 40 + "\n")
            
            comment_intervals = convert_to_intervals([int(x) for x in comment_indices])
            code_intervals = convert_to_intervals([int(x) for x in code_indices])

            # log_file.write(f"Intervals - Comment: {comment_intervals}, Code: {code_intervals}\n")
            
            if len(comment_intervals) > 0 and len(code_intervals) > 0:
                final_results.append([comment_intervals, code_intervals])

        if not final_results:
            log_file.write(f"No final results for auto_label_ind: {auto_label_ind}\n")
            continue

        log_file.write(f"Final results: {final_results}\n")

        new_entry = {
            "idx": int(student_idx),
            "match": final_results
        }

        file_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/label_human_auto_index.jsonl"
        with open(file_path, 'a') as file:
            file.write(json.dumps(new_entry) + '\n')
            log_file.write("Successfully wrote new entry to JSONL file\n")