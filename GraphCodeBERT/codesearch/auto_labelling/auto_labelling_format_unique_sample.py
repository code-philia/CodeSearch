import os
import re
from openai import OpenAI
import json
import wordninja
import random
import numpy as np
import time


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
    
    # Track function name after def
    function_name = None
    
    for i, token in enumerate(tokens):
        # Skip comment tokens starting with #
        if token.startswith('#'):
            continue
            
        # Track starting index for this original token
        start_idx = result_idx
        
        # If previous token was 'def', store current token as function name
        prev_token = tokens[i-1] if i > 0 else None
        if prev_token == 'def':
            function_name = token
        
        # If token is a single special character, add it directly
        if len(token) == 1 and not (token.isalnum() or token == '_'):
            # Special case: if this is a * and previous token was also *, reuse previous map_idx
            if token == '*' and result and result[-1] == '*':
                if map_idx > 0:  # Make sure we have a previous index to reference
                    token_map[map_idx-1].append(result_idx)
                    result.append(token)
                    result_idx += 1
                    continue
            
            result.append(token)
            token_map[map_idx] = [result_idx]
            result_idx += 1
            map_idx += 1
            continue
            
        # Check if token should be kept intact (API call or function name)
        prev_token = tokens[i-1] if i > 0 else None
        next_token = tokens[i+1] if i < len(tokens)-1 else None
        
        keep_intact = False
        if (prev_token == '.' or next_token == '.' or next_token == '('):
            # Exception: split if this is a function definition name or matches function name
            if prev_token != 'def' and token != function_name:
                keep_intact = True
                
        if keep_intact:
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
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = "sk-WncqiWssXTIYL9PpaL5t2YxEObWkZvlxeOj8o45uo64lvQoP"
os.environ["OPENAI_BASE_URL"] = "https://api.key77qiqi.cn/v1"

# load all training data
# 文件路径
file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/train.jsonl'

# 存储所有数据的列表
train_data = []
with open(file_path, 'r') as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

# load all training data tokens
file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/tokenized_code_tokens_train.jsonl'
code_tokens_strs = []
with open(file_path, 'r') as f:
    for line in f:
        code_tokens_strs.append(json.loads(line.strip()))

nl_file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/tokenized_comment_tokens_train.jsonl'
nl_tokens_strs = []
with open(nl_file_path, 'r') as f:
    for line in f:
        nl_tokens_strs.append(json.loads(line.strip()))

# 现在 code_tokens_strs 变量中包含了从 JSON 文件读取的数据
print("len(code_tokens_strs)", len(code_tokens_strs))  # 可以查看加载的数据
print("len(nl_tokens_strs)", len(nl_tokens_strs))  # 可以查看加载的数据

# Further tokenize code tokens and docstring tokens
# Process all training data with token mapping
tokens_output_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/code_tokens_further.jsonl"
maps_output_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/code_token_maps.jsonl"

# Check if both files exist
if os.path.exists(tokens_output_path) and os.path.exists(maps_output_path):
    print("Found existing tokenized files, loading them...")
    code_tokens_further_data = []
    code_token_maps = []
    
    # Load existing tokens
    with open(tokens_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            code_tokens_further_data.append(json_obj['code_tokens_further'])
            
    # Load existing maps
    with open(maps_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            code_token_maps.append(json_obj['code_token_map'])
            
else:
    print("Tokenized files not found, generating them...")
    code_tokens_further_data = []
    code_token_maps = []

    # Process whole data
    for data in train_data:
        code_tokens = data['code_tokens']
        further_tokens, token_mapping = further_tokenize(code_tokens)
        code_tokens_further_data.append(further_tokens)
        code_token_maps.append(token_mapping)

    # Save further tokens
    with open(tokens_output_path, 'w', encoding='utf-8') as f:
        for tokens in code_tokens_further_data:
            json_obj = {'code_tokens_further': tokens}
            f.write(json.dumps(json_obj) + '\n')
            

    # Save token mappings 
    with open(maps_output_path, 'w', encoding='utf-8') as f:
        for token_map in code_token_maps:
            json_obj = {'code_token_map': token_map}
            f.write(json.dumps(json_obj) + '\n')

# load already auto labeled info
input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_teacher.jsonl"
idx_list = []
match_list = []

with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().rstrip(',')  # 去除行末的逗号
        json_obj = json.loads(line)
        idx_list.append(json_obj['idx'])
        match_list.append(json_obj['match'])

print("len(idx_list)", len(idx_list)) 


# load map data
with open(maps_output_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        code_token_maps.append(json_obj['code_token_map'])

# load tokenize id data
with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/tokenized_id_train.json', 'r') as f:
    tokenized_id_data = json.load(f)
print("len(tokenized_id_data)", len(tokenized_id_data))     

# Load or generate student-teacher pairs based on similarity
student_teacher_pairs_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/student_teachers_pairs.jsonl"

if os.path.exists(student_teacher_pairs_file):
    student_teacher_pairs = []
    with open(student_teacher_pairs_file, 'r') as f:
        for line in f:
            student_teacher_pairs.append(json.loads(line))
    print(f"Loaded {len(student_teacher_pairs)} existing student-teacher pairs")

else:
    # Load grammar patterns
    with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/core_grammar_stats.json', 'r') as f:
        grammar_stats = json.load(f)
        sentence_patterns = grammar_stats['sentence_patterns']

    ast_vectors = np.load('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/scaled_ast_vectors.npy')
    # Process each training example
    student_teacher_pairs = []
    total = len(ast_vectors)
    
    print(f"Starting to process {total} examples...")
    for student_idx in range(total):
        if student_idx % 1000 == 0:
            print(f"Processing example {student_idx}/{total} ({(student_idx/total)*100:.1f}%)")
            
        if student_idx in idx_list:
            continue
            
        student_pattern = sentence_patterns[str(student_idx)]
        student_vector = ast_vectors[student_idx]
        
        # Find teachers with same pattern
        similarities = []
        for teacher_idx in idx_list:
            teacher_pattern = sentence_patterns[str(teacher_idx)]
            
            # Only consider teachers with same pattern
            if teacher_pattern != student_pattern:
                continue
                
            teacher_vector = ast_vectors[teacher_idx]
            
            # Calculate cosine similarity between AST vectors
            similarity = np.dot(student_vector, teacher_vector) / (np.linalg.norm(student_vector) * np.linalg.norm(teacher_vector))
            
            similarities.append({
                'teacher_idx': teacher_idx,
                'confidence': similarity
            })
        
        # Only proceed if we found at least 3 teachers
        if len(similarities) >= 3:
            # Sort by similarity and take top 3
            similarities.sort(key=lambda x: x['confidence'], reverse=True)
            top_3_teachers = similarities[:3]
            
            pair = {
                'student_idx': student_idx,
                'teachers': top_3_teachers
            }
            # Write each pair to jsonl file
            with open(student_teacher_pairs_file, 'a') as f:
                f.write(json.dumps(pair) + '\n')
            student_teacher_pairs.append(pair)

    print(f"Completed! Generated and saved {len(student_teacher_pairs)} student-teacher pairs based on token similarity")


# Process all training data and save results
print("Processing training data with unique symbols/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling.")

# Save processed docstring tokens to file
docstring_tokens_unique_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/docstring_tokens_unique.jsonl"
if not os.path.exists(docstring_tokens_unique_file):
    with open(docstring_tokens_unique_file, 'w') as f:
        for data in train_data:
            data['docstring_tokens_unique'] = add_unique_symbols(data['docstring_tokens'], is_comment=True)
            f.write(json.dumps({'docstring_tokens_unique': data['docstring_tokens_unique']}) + '\n')
else:
    with open(docstring_tokens_unique_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            train_data[i]['docstring_tokens_unique'] = data['docstring_tokens_unique']

# Save processed code tokens to file  
code_tokens_unique_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/code_tokens_unique.jsonl"
if not os.path.exists(code_tokens_unique_file):
    with open(code_tokens_unique_file, 'w') as f:
        for tokens in code_tokens_further_data:
            unique_tokens = add_unique_symbols(tokens, is_comment=False)
            f.write(json.dumps({'code_tokens_unique': unique_tokens}) + '\n')
else:
    code_tokens_further_data_unique = []
    with open(code_tokens_unique_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            code_tokens_further_data_unique.append(data['code_tokens_unique'])

print("Finished loading/processing unique symbols for tokens")

# Set confidence thresholds
MEAN_CONF_THRESHOLD = 0.65  # Minimum required mean confidence
LOWEST_CONF_THRESHOLD = 0.55  # Minimum required lowest confidence

# Filter student-teacher pairs based on confidence thresholds
auto_label_indices = []
for idx, pair in enumerate(student_teacher_pairs):
    # Get lowest confidence among teachers
    lowest_conf = min(teacher['confidence'] for teacher in pair['teachers'])
    
    # Only keep pairs that meet both thresholds
    mean_conf = sum(teacher['confidence'] for teacher in pair['teachers']) / len(pair['teachers'])
    if mean_conf >= MEAN_CONF_THRESHOLD and lowest_conf >= LOWEST_CONF_THRESHOLD:
        auto_label_indices.append(idx)

print(f"Selected {len(auto_label_indices)} samples that meet confidence thresholds")
print(f"Mean confidence threshold: {MEAN_CONF_THRESHOLD}")
print(f"Lowest confidence threshold: {LOWEST_CONF_THRESHOLD}")

# Load previously labeled data to avoid duplicates
labeled_indices = set()
try:
    with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_2.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line)
            labeled_indices.add(entry['idx'])
except FileNotFoundError:
    print("No existing labeled data file found, starting fresh")

# Filter out already labeled examples and shuffle remaining indices
auto_label_indices = [idx for idx in auto_label_indices 
                     if student_teacher_pairs[idx]['student_idx'] not in labeled_indices]
random.shuffle(auto_label_indices)

# Only process first 1000 examples
auto_label_indices = auto_label_indices[:8000]


print(f"After filtering already labeled examples, {len(auto_label_indices)} samples remain")


system_prompt = "You are an expert at aligning tokens between comments and code. You can accurately identify the similarities and differences between tokens, and you are highly skilled at matching tokens based on their semantics and functionality. You are given input data consisting of comment tokens and code tokens, and your task is to align them by identifying concepts in the comments and matching them to corresponding code tokens. Use the example cases below and output your results in the specified format."

# auto labelling 
# Only process first 10 examples for testing
for auto_label_ind in auto_label_indices:
    # Get student and teacher indices from the current pair
    student_teacher_pair = student_teacher_pairs[auto_label_ind]
    student_idx = student_teacher_pair['student_idx']
    teachers = student_teacher_pair['teachers']

    # Initialize list to store all teacher examples
    teacher_examples = []

    # Process each teacher
    for teacher in teachers:
        # Get the actual teacher index from the dictionary
        teacher_idx = teacher['teacher_idx']
        
        # Get unique symbolized tokens for docstring and code
        unique_docstring_tokens = train_data[teacher_idx]['docstring_tokens_unique']
        unique_code_tokens = code_tokens_further_data_unique[teacher_idx]
        
        # Initialize list to store all alignments for this teacher
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

        # Create teacher example for current teacher
        teacher_example = f"""
        **Teacher Example {len(teacher_examples) + 1}:**
        Comment Tokens Index and Comment Tokens String:
        {unique_docstring_tokens}
        Code Tokens Index and Code Tokens String:
        {unique_code_tokens}
        **Matching Output:**
        {alignments}
        """
        teacher_examples.append(teacher_example)

    # Combine all teacher examples into one prompt
    teacher_prompt = "\n".join(teacher_examples)

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
    with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/auto_labelling_teachers_confidence_2.log', 'a') as log_file:
        log_file.write(f"\n=== Processing auto_label_ind: {auto_label_ind} ===\n")
        for i, teacher in enumerate(teachers):
            log_file.write(f"Teacher {i} idx: {teacher['teacher_idx']}, confidence: {teacher['confidence']}\n")
        log_file.write(f"Student idx: {student_idx}\n")
        log_file.write(f"Student tokens: {student_tokens_part}\n")

        try:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
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
        except Exception as e:
            log_file.write(f"API call error: {str(e)}\n")
            log_file.write("Pausing for 2 minutes before continuing...\n")
            time.sleep(120)  # 暂停2分钟
            continue

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

        # Initialize sets to track matched indices
        matched_comment_indices = set()
        matched_code_indices = set()
        has_hallucination = False # Track hallucination at global level
        
        # Track matched tokens per alignment round
        matched_token_texts = {} # Dict mapping alignment index to set of token texts
        
        for alignment_idx, alignment in enumerate(alignment_output["alignments"]):
            comment_matches = []
            code_matches = []
            code_current_pos = 0
            comment_current_pos = 0
            
            # Initialize set for this alignment round
            matched_token_texts[alignment_idx] = set()
            
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
            
            # Process code tokens with new matching logic
            log_file.write("\nProcessing code tokens:\n")
            
            for code_token in alignment["code_token"]:
                # Check for hallucination by looking up token in unique tokens
                try:
                    token_idx = code_tokens_further_data_unique[student_idx].index(code_token)
                    log_file.write(f"Found token {code_token} at index {token_idx}\n")
                except ValueError:
                    log_file.write(f"Could not find token {code_token} in unique tokens - hallucination detected\n")
                    has_hallucination = True
                    break
                    
                # Find which original token this was derived from using code_token_maps
                for orig_idx, derived_indices in code_token_maps[student_idx].items():
                    if token_idx in derived_indices:
                        # Convert orig_idx from string to integer before using it
                        orig_idx_int = int(orig_idx)
                        log_file.write(f"Original key {orig_idx} was converted to integer {orig_idx_int}\n")
                        # Get the tokenized range for this original token
                        try:
                            tokenize_range = tokenized_id_data[student_idx][1:][orig_idx_int]
                        except IndexError:
                            continue
                        log_file.write(f"Original token at index {orig_idx} was tokenized to range {tokenize_range}\n")
                        
                        # Get the clean version of the unique token
                        clean_token = re.sub(r'[▲■].*$', '', code_token)
                        
                        # Try to match the clean token against the tokenized pieces
                        start_idx = tokenize_range[0]  # Keep original [a,b] range
                        end_idx = tokenize_range[1] + 1
                        
                        # Get the tokens in the range, including start index
                        range_tokens = code_tokens_strs[student_idx][start_idx+1:end_idx]
                        log_file.write(f"Range tokens: {range_tokens}\n")
                        
                        # Try to find minimal token combinations that match clean_token
                        i = 0
                        found_any_match = False
                        while i < len(range_tokens):
                            # Try combining minimal number of tokens starting at position i
                            min_combined = ""
                            j = i
                            min_matched_indices = []
                            found_match = False
                            
                            while j < len(range_tokens):
                                curr_token = range_tokens[j].replace('Ġ', '')
                                min_combined += curr_token
                                min_matched_indices.extend(range(start_idx+i, start_idx+j+1))
                                
                                # Check for exact match with minimal tokens
                                if min_combined == clean_token:
                                    indices_range = set(min_matched_indices)
                                    # Only match if tokens haven't been used before
                                    if not indices_range.intersection(matched_code_indices):
                                        code_matches.extend(list(indices_range))  # Convert to list to extend
                                        matched_code_indices.update(indices_range)
                                        # Add matched text to current alignment's set
                                        matched_token_texts[alignment_idx].add(min_combined)
                                        log_file.write(f"Added minimal indices {list(indices_range)} for combined token {min_combined}\n")
                                        found_match = True
                                        found_any_match = True
                                        break
                                # Stop if combined tokens exceed clean_token length
                                elif len(min_combined) > len(clean_token):
                                    break
                                    
                                j += 1
                                
                            if found_match:
                                break
                            i += 1

                        # If no matches found for any token in this range, match all range tokens to first clean token
                        if not found_any_match:
                            all_indices = list(range(start_idx, end_idx-1))
                            available_indices = [i for i in all_indices if i not in matched_code_indices]
                            if available_indices:
                                code_matches.extend(available_indices)
                                matched_code_indices.update(available_indices)
                                combined_token = ''.join([t.replace('Ġ', '') for t in range_tokens])
                                matched_token_texts[alignment_idx].add(combined_token)
                                log_file.write(f"No individual matches found - matching available indices {available_indices} for combined token {combined_token} to {clean_token}\n")
                        break
            
            # If hallucination detected, skip this alignment
            if has_hallucination:
                log_file.write("Hallucination detected - skipping all alignments\n")
                final_results = [] # Clear all previous results
                break
                    
            log_file.write(f"Final code matches: {code_matches}\n")
            
            # Keep indices for now
            if len(comment_matches) > 0 and len(code_matches) > 0:
                final_results.append([comment_matches, code_matches])
                log_file.write(f"Added to final results: {[comment_matches, code_matches]}\n")

        # Post-validation: Find identical unmatched tokens
        for i in range(len(code_tokens_strs[student_idx])):
            if i-1 not in matched_code_indices:
                # Get the token text and skip special tokens
                token_text = code_tokens_strs[student_idx][i].replace('Ġ', '')
                # Skip common special characters/operators
                if token_text in ['.', '_', ':', ';', ',', '(', ')', '[', ']', '{', '}', 
                                '+', '-', '*', '/', '=', '<', '>', '!', '&', '|', '^',
                                'self', 'def', '<unk>']:
                    continue
                    
                # Check if token appears in any previous alignment rounds
                appearing_rounds = []
                for round_idx, token_set in matched_token_texts.items():
                    if token_text in token_set:
                        appearing_rounds.append(round_idx)
                
                if appearing_rounds:
                    if len(appearing_rounds) > 1:
                        # Token appears in multiple rounds - invalidate results
                        log_file.write(f"Token {token_text} appears in multiple rounds {appearing_rounds} - invalidating results\n")
                        break
                    else:
                        # Token appears in exactly one round - add to that round's matches
                        round_idx = appearing_rounds[0]
                        
                        # Check if round_idx is valid
                        if round_idx >= len(final_results):
                            log_file.write(f"Invalid round index {round_idx} - skipping token {token_text}\n")
                            continue
                            
                        # Get the existing indices for this round
                        try:
                            comment_indices = final_results[round_idx][0]
                            code_indices = final_results[round_idx][1]
                        except IndexError:
                            log_file.write(f"Invalid final results format for round {round_idx} - skipping token {token_text}\n")
                            continue
                        
                        # Add the new code index
                        code_indices.append(i-1)
                        
                        # Update the final results with new code indices
                        final_results[round_idx] = [comment_indices, code_indices]
                        
                        matched_code_indices.add(i-1)
                        log_file.write(f"Post-validation: Added identical unmatched token at index {i-1}: {token_text} to round {round_idx}\n")
                        
        # Convert indices to intervals for each round at the end
        final_intervals = []
        for comment_indices, code_indices in final_results:
            comment_intervals = convert_to_intervals(sorted(comment_indices))
            code_intervals = convert_to_intervals(sorted(code_indices))
            final_intervals.append([comment_intervals, code_intervals])
            log_file.write(f"\nConverted to intervals:\nComment intervals: {comment_intervals}\nCode intervals: {code_intervals}\n")
        
        final_results = final_intervals

        if not final_results:
            log_file.write(f"No final results for auto_label_ind: {auto_label_ind}\n")
            continue

        log_file.write(f"Final results: {final_results}\n")

        new_entry = {
            "idx": int(student_idx),
            "match": final_results
        }

        file_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api_student_conf_2.jsonl"
        with open(file_path, 'a') as file:
            file.write(json.dumps(new_entry, separators=(',',':'), ensure_ascii=False) + '\n')
            log_file.write("Successfully wrote new entry to JSONL file\n")