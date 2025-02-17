import os
import re
from openai import OpenAI
import json
import wordninja
import random


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

# Function to further tokenize code tokens and track token mappings
def further_tokenize(tokens):
    result = []
    token_map = {}  # Map non-comment token index to list of split token indices
    result_idx = 0
    map_idx = 0  # Separate counter for non-comment tokens
    
    for i, token in enumerate(tokens):
        # Skip comment tokens starting with #
        if token.startswith('#'):
            continue
            
        # Track starting index for this original token
        start_idx = result_idx
        
        # If token is a single special character, add it directly
        if len(token) == 1 and not (token.isalnum() or token == '_'):
            result.append(token)
            token_map[map_idx] = [result_idx]
            result_idx += 1
            map_idx += 1
            continue
            
        # Split by special characters while keeping them
        parts = []
        current = ''
        for char in token:
            if char.isalnum() or char == '_':
                current += char
            else:
                if current:
                    parts.append(current)
                    current = ''
                if char != ' ':  # Keep all special characters except spaces
                    parts.append(char)
        if current:
            parts.append(current)
            
        # Process each part
        for part in parts:
            if '_' in part:
                # Handle underscore-separated parts
                if part.startswith('_'):
                    result.append('_')
                    result_idx += 1
                    
                subparts = [p for p in part.split('_') if p != '']
                
                for j, subpart in enumerate(subparts):
                    if j > 0:
                        result.append('_')
                        result_idx += 1
                    if subpart.strip('_'):
                        split_subparts = wordninja.split(subpart)
                        result.extend(p for p in split_subparts if p)
                        result_idx += len([p for p in split_subparts if p])
                
                if part.endswith('_'):
                    result.append('_')
                    result_idx += 1
            else:
                # Handle other parts
                split_parts = wordninja.split(part)
                result.extend(p for p in split_parts if p)
                result_idx += len([p for p in split_parts if p])
                
        # Map non-comment token to range of split tokens
        token_map[map_idx] = list(range(start_idx, result_idx))
        map_idx += 1
                    
    return result, token_map

# Function to add unique symbols to repeated tokens
def add_unique_symbols(tokens, is_comment=True):
    # Track token counts and processed tokens
    token_counts = {}
    processed_tokens = []
    
    # First pass - count occurrences
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
            
    # Second pass - add symbols to repeated tokens
    token_seen = {}
    for token in tokens:
        if token_counts[token] > 1:
            # Track occurrence number for this token
            if token not in token_seen:
                token_seen[token] = 1
            else:
                token_seen[token] += 1
                
            # Add triangle (▲) for comments, square (■) for code
            symbol = "▲" if is_comment else "■"
            processed_tokens.append(f"{token}{symbol}{token_seen[token]}")
        else:
            processed_tokens.append(token)
            
    return processed_tokens

# set openai environ and key
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = "sk-jk4LBMpM4HzVEVYrbv3mS0gbl6kQskVQnXlqY3KlqS5o9lbY"
os.environ["OPENAI_BASE_URL"] = "https://api.key77qiqi.cn/v1"

# load all training data
# 文件路径
file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/dataset/python/train.jsonl'

# 存储所有数据的列表
train_data = []
with open(file_path, 'r') as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

# load all training data tokens
file_path = '/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_code_tokens_train.json'
with open(file_path, 'r') as f:
    code_tokens_strs = json.load(f)

nl_file_path = '/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_comment_tokens_train.json'
with open(nl_file_path, 'r') as f:
    nl_tokens_strs = json.load(f)

# 现在 code_tokens_strs 变量中包含了从 JSON 文件读取的数据
print("len(code_tokens_strs)", len(code_tokens_strs))  # 可以查看加载的数据
print("len(nl_tokens_strs)", len(nl_tokens_strs))  # 可以查看加载的数据

# Further tokenize code tokens and docstring tokens
# Process all training data with token mapping
code_tokens_further = []
code_token_maps = []

# for data in train_data:
#     code_tokens = data['code_tokens']
#     further_tokens, token_mapping = further_tokenize(code_tokens)
#     code_tokens_further.append(further_tokens)
#     code_token_maps.append(token_mapping)

# Save code_tokens_further to a jsonl file
tokens_output_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/code_tokens_further.jsonl"
maps_output_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/code_token_maps.jsonl"

# # Save further tokens
# with open(tokens_output_path, 'w', encoding='utf-8') as f:
#     for tokens in code_tokens_further:
#         json_obj = {'code_tokens_further': tokens}
#         f.write(json.dumps(json_obj) + '\n')

# # Save token mappings 
# with open(maps_output_path, 'w', encoding='utf-8') as f:
#     for token_map in code_token_maps:
#         json_obj = {'code_token_map': token_map}
#         f.write(json.dumps(json_obj) + '\n')

# Load code_tokens_further from jsonl file
code_tokens_further_data = []
with open(tokens_output_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        code_tokens_further_data.append(json_obj['code_tokens_further'])

# load already auto labeled info
input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/sorted_labelling_sample_api.jsonl"
idx_list = []
match_list = []

with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().rstrip(',')  # 去除行末的逗号
        json_obj = json.loads(line)
        idx_list.append(json_obj['idx'])
        match_list.append(json_obj['match'])

print("len(idx_list)", len(idx_list)) 

unlabeled_indices = list(set(range(len(train_data))) - set(idx_list)) 
print("len(unlabeled_indices)", len(unlabeled_indices)) 

# load map data
with open(maps_output_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        code_token_maps.append(json_obj['code_token_map'])

# load tokenize id data
with open('/home/yiming/cophi/training_dynamic/features/tokenized_id_train.json', 'r') as f:
    tokenized_id_data = json.load(f)

# Initialize lists to store student-teacher pairs
student_teacher_pairs = []

# Set similarity threshold
SIMILARITY_THRESHOLD = 0.8

# For each unlabeled index (potential student)
for student_idx in unlabeled_indices:
    student_tokens = code_tokens_further_data[student_idx]  # Use loaded data instead
    student_token_set = set(student_tokens)
    
    max_similarity = 0
    best_teacher = None
    
    # Compare with each auto-labeled index (potential teacher)
    for teacher_idx in idx_list:
        teacher_tokens = code_tokens_further_data[teacher_idx]  # Use loaded data instead
        teacher_token_set = set(teacher_tokens)
        
        # Calculate token overlap
        common_tokens = student_token_set.intersection(teacher_token_set)
        similarity = len(common_tokens) / len(student_token_set)
        
        # Update best match if current similarity is higher
        if similarity > max_similarity:
            max_similarity = similarity
            best_teacher = teacher_idx
    
    # If similarity exceeds threshold, add to pairs
    if max_similarity >= SIMILARITY_THRESHOLD:
        student_teacher_pairs.append({
            'student_idx': student_idx,
            'teacher_idx': best_teacher,
            'similarity': max_similarity
        })

print(f"Found {len(student_teacher_pairs)} student-teacher pairs based on token similarity")

# Process all training data and save results
print("Processing training data with unique symbols...")

# Save processed docstring tokens to file
docstring_tokens_unique_file = "./docstring_tokens_unique.json"
if not os.path.exists(docstring_tokens_unique_file):
    for data in train_data:
        data['docstring_tokens_unique'] = add_unique_symbols(data['docstring_tokens'], is_comment=True)
    with open(docstring_tokens_unique_file, 'w') as f:
        json.dump([data['docstring_tokens_unique'] for data in train_data], f)
else:
    with open(docstring_tokens_unique_file, 'r') as f:
        docstring_tokens_unique = json.load(f)
        for i, data in enumerate(train_data):
            data['docstring_tokens_unique'] = docstring_tokens_unique[i]

# Save processed code tokens to file  
code_tokens_unique_file = "./code_tokens_unique.json"
if not os.path.exists(code_tokens_unique_file):
    code_tokens_further_data_unique = []
    for tokens in code_tokens_further_data:
        unique_tokens = add_unique_symbols(tokens, is_comment=False)
        code_tokens_further_data_unique.append(unique_tokens)
    with open(code_tokens_unique_file, 'w') as f:
        json.dump(code_tokens_further_data_unique, f)
else:
    with open(code_tokens_unique_file, 'r') as f:
        code_tokens_further_data_unique = json.load(f)

print("Finished loading/processing unique symbols for tokens")


# Shuffle the student-teacher pairs for random processing order
random.shuffle(student_teacher_pairs)

system_prompt = "You are an expert at aligning tokens between comments and code. You can accurately identify the similarities and differences between tokens, and you are highly skilled at matching tokens based on their semantics and functionality. You are given input data consisting of comment tokens and code tokens, and your task is to align them by identifying concepts in the comments and matching them to corresponding code tokens. Use the example cases below and output your results in the specified format."

# auto labelling 
# for i in range(len(auto_label_indices)):
for auto_label_ind in range(1):
    # Get student and teacher indices from the current pair
    student_teacher_pair = student_teacher_pairs[auto_label_ind]
    student_idx = student_teacher_pair['student_idx']
    teacher_idx = student_teacher_pair['teacher_idx']
    similarity = student_teacher_pair['similarity']

    # Get unique symbolized tokens for docstring and code
    unique_docstring_tokens = train_data[teacher_idx]['docstring_tokens_unique']
    unique_code_tokens = code_tokens_further_data_unique[teacher_idx]
    
    # Initialize list to store all alignments for this teacher_idx
    alignments = []

    # Find the index in idx list that matches teacher_idx
    teacher_ind = None
    for i, entry in enumerate(idx_list):
        if entry == teacher_idx:
            teacher_ind = i
            break
            
    if teacher_ind is None:
        print(f"Could not find teacher_idx {teacher_idx} in idx_list")
        continue

    # For each match pair in the match list
    for match_pair in match_list[teacher_ind]:
        # Get comment and code match indices
        comment_match = match_pair[0]
        code_match = match_pair[1]
        
        # Initialize lists for current pair's tokens
        current_comment_tokens = []
        current_code_tokens = []
        
        # Extract matched comment tokens using indices
        for i in range(0, len(comment_match), 2):
            start, end = comment_match[i], comment_match[i+1]
            tokens = nl_tokens_strs[teacher_idx][1:][start:end+1]
            for token in tokens:
                clean_token = token.replace('Ġ', '')
                current_comment_tokens.append(clean_token)
            
        # Extract matched code tokens using indices  
        for i in range(0, len(code_match), 2):
            start, end = code_match[i], code_match[i+1]
            tokens = code_tokens_strs[teacher_idx][1:][start:end+1]
            for token in tokens:
                clean_token = token.replace('Ġ', '')
                current_code_tokens.append(clean_token)

        # Map current pair's tokens to new token lists
        docstring_tokens = train_data[teacher_idx]['docstring_tokens']
        code_tokens = code_tokens_further_data[teacher_idx]
        
        # Create mappings for current pair
        current_comment_map = {}
        current_code_map = {}
        
        # Map comment tokens for current pair
        doc_start_idx = 0
        for i, matched_token in enumerate(current_comment_tokens):
            for j in range(doc_start_idx, len(docstring_tokens)):
                if matched_token in docstring_tokens[j]:
                    current_comment_map[f"{i}_{matched_token}"] = f"{j}_{docstring_tokens[j]}"
                    doc_start_idx = j + 1
                    break
            
        # Map code tokens for current pair
        code_start_idx = 0
        for i, matched_token in enumerate(current_code_tokens):
            for j in range(code_start_idx, len(code_tokens)):
                if matched_token in code_tokens[j]:
                    current_code_map[f"{i}_{matched_token}"] = f"{j}_{code_tokens[j]}"
                    code_start_idx = j + 1
                    break

        # Extract indices from current mappings
        comment_indices = [int(v.split('_')[0]) for v in current_comment_map.values()]
        code_indices = [int(v.split('_')[0]) for v in current_code_map.values()]

        # Get corresponding unique symbolized tokens
        comment_unique_tokens = [unique_docstring_tokens[idx] for idx in comment_indices]
        code_unique_tokens = [unique_code_tokens[idx] for idx in code_indices]

        # Create alignment dict for current pair
        alignment = {
            "comment_token": comment_unique_tokens,
            "code_token": code_unique_tokens
        }
        
        # Add to alignments list
        alignments.append(alignment)

    # construct teacher_prompt
    teacher_prompt = f"""
    Below is an example that demonstrates how to align comment tokens and code tokens:
    **Teacher Example:**
    Comment Tokens Index and Comment Tokens String:
    {unique_docstring_tokens}
    Code Tokens Index and Code Tokens String:
    {unique_code_tokens}
    **Matching Output:**
    {alignments}
    """

    # construct student input with indices
    unique_student_docstring_tokens = train_data[student_idx]['docstring_tokens_unique']
    unique_student_code_tokens = code_tokens_further_data_unique[student_idx]

    student_docstring_tokens_raw = nl_tokens_strs[student_idx][1:]
    student_code_tokens_raw = code_tokens_strs[student_idx][1:]

    # construct student_prompt
    student_tokens_part = f"""
    Here are the tokens you need to process:

    Comment Tokens Index and Comment Tokens String:
    {unique_student_docstring_tokens}

    Code Tokens Index and Code Tokens String:
    {unique_student_code_tokens}
    """

    alignment_format = """
    {
        "alignments": [
            {"comment_token": ["token1", "token2"], "code_token": ["tokenA", "tokenB"]},
            {"comment_token": ["token3", "token4"], "code_token": ["tokenC", "tokenD"]},
            {"comment_token": ["token5", "token6"], "code_token": ["tokenE", "tokenF"]}
        ]
    }
    """

    student_prompt = f"""
    CRITICAL ALIGNMENT INSTRUCTIONS:
    1. FIRST analyze comment token types:
       - Identify VERBS (actions, operations)
       - Identify NOUNS (objects, concepts)
       - Keep these categories separate
       - Each comment token can ONLY be used ONCE in matching
    2. THEN categorize code tokens:
       - FUNCTION NAMES (methods, routines)
       - VARIABLE NAMES (parameters, fields)
       - API CALLS (library functions)
       - KEYWORDS (control flow, operators)
       - Each code token can ONLY be matched ONCE
    3. ENFORCE concept extraction rules:
       - Extract concepts ONLY from comment VERBS and NOUNS
       - Each comment token can only form ONE concept
       - Ignore modifiers and other parts of speech
       - Ensure concepts are semantically meaningful
    4. Follow matching principles:
       - Each code token can only match ONE concept
       - Match concepts to most semantically similar code tokens
    5. MAXIMIZE valid concept-code matches:
       - Choose strongest semantic matches between concepts and code
       - Leave concepts/tokens unmatched rather than force weak matches

    Here are the tokens to align:
    {student_tokens_part}

    Based on the above instructions and following the teacher example, provide comprehensive alignments between comment concepts and code implementations. Output in this format:
    {alignment_format}
    """

    promt_str = system_prompt + teacher_prompt + student_prompt

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))

    # 打开日志文件
    with open('./auto_labelling.log', 'a') as log_file:
        log_file.write(f"\n=== Processing auto_label_ind: {auto_label_ind} ===\n")
        log_file.write(f"Teacher idx: {teacher_idx}\n")
        log_file.write(f"Student idx: {student_idx}\n")
        log_file.write(f"Similarity: {similarity}\n")

        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": promt_str}],
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "strict": True,
                    "name": "alignment_response",
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
                                            "items": {"type": "string"}
                                        },
                                        "code_token": {
                                            "type": "array", 
                                            "items": {"type": "string"}
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

        # Initialize sets to track matched indices
        matched_comment_indices = set()
        matched_code_indices = set()
        
        for alignment in alignment_output["alignments"]:
            comment_matches = []
            code_matches = []
            code_current_pos = 0
            comment_current_pos = 0
            
            # Process comment tokens
            log_file.write("\nProcessing comment tokens:\n")
            for token in alignment["comment_token"]:
                # Remove special characters (triangles and squares) from token
                clean_token = re.sub(r'[▲■].*$', '', token)
                log_file.write(f"Original token: {token}, Cleaned token: {clean_token}\n")
                
                # Find first valid continuous match starting from last match position
                comment_token_indices = []
                last_match_pos = comment_current_pos if comment_current_pos > 0 else 0
                
                # Try to find first valid continuous match starting from last match position
                i = last_match_pos
                found_match = False
                while i < len(student_docstring_tokens_raw) and not found_match:
                    # Get continuous tokens starting at position i
                    continuous_token = ""
                    j = i
                    while j < len(student_docstring_tokens_raw):
                        continuous_token += student_docstring_tokens_raw[j].replace('Ġ', '')
                        if clean_token == continuous_token:
                            # Check if any index in range is already matched
                            indices_range = set(range(i, j + 1))
                            if not indices_range.intersection(matched_comment_indices):
                                comment_token_indices.extend(range(i, j + 1))
                                comment_current_pos = j + 1
                                matched_comment_indices.update(indices_range)
                                log_file.write(f"Match found from index {i} to {j}\n")
                                found_match = True
                            break
                        j += 1
                    if not found_match:
                        i += 1
                    
                comment_matches.extend(comment_token_indices)
            log_file.write(f"Final comment matches: {comment_matches}\n")
            
            # Process code tokens  
            log_file.write("\nProcessing code tokens:\n")
            for code_token in alignment["code_token"]:
                # Remove special characters from token
                clean_token = re.sub(r'[▲■].*$', '', code_token)
                log_file.write(f"Original token: {code_token}, Cleaned token: {clean_token}\n")
                
                # Find first valid continuous match starting from last match position
                code_token_indices = []
                last_match_pos = code_current_pos if code_current_pos > 0 else 0
                
                # Try to find first valid continuous match starting from last match position
                i = last_match_pos
                found_match = False
                while i < len(student_code_tokens_raw) and not found_match:
                    # Get continuous tokens starting at position i
                    continuous_token = ""
                    j = i
                    while j < len(student_code_tokens_raw):
                        continuous_token += student_code_tokens_raw[j].replace('Ġ', '')
                        if clean_token == continuous_token:
                            # Check if any index in range is already matched
                            indices_range = set(range(i, j + 1))
                            if not indices_range.intersection(matched_code_indices):
                                code_token_indices.extend(range(i, j + 1))
                                code_current_pos = j + 1
                                matched_code_indices.update(indices_range)
                                log_file.write(f"Match found from index {i} to {j}\n")
                                found_match = True
                            break
                        j += 1
                    if not found_match:
                        i += 1
                    
                code_matches.extend(code_token_indices)
            log_file.write(f"Final code matches: {code_matches}\n")
            
            # Convert indices to intervals
            comment_intervals = convert_to_intervals(sorted(comment_matches))
            code_intervals = convert_to_intervals(sorted(code_matches))
            log_file.write(f"\nConverted to intervals:\nComment intervals: {comment_intervals}\nCode intervals: {code_intervals}\n")
            
            if len(comment_intervals) > 0 and len(code_intervals) > 0:
                final_results.append([comment_intervals, code_intervals])
                log_file.write(f"Added to final results: {[comment_intervals, code_intervals]}\n")

        if not final_results:
            log_file.write(f"No final results for auto_label_ind: {auto_label_ind}\n")
            continue

        log_file.write(f"Final results: {final_results}\n")

        new_entry = {
            "idx": int(student_idx),
            "match": final_results
        }

        file_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_label_unique.jsonl"
        with open(file_path, 'a') as file:
            file.write(json.dumps(new_entry) + '\n')
            log_file.write("Successfully wrote new entry to JSONL file\n")