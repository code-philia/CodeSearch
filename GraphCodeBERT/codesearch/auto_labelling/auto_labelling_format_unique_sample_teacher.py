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
os.environ["OPENAI_API_KEY"] = "sk-5QMoD9yjiyQb8hQcymsFeKhvh0rhkdJ1GiwRknz0rxtVgZg0"
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
file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/tokenized_code_tokens_train.json'
with open(file_path, 'r') as f:
    code_tokens_strs = json.load(f)

nl_file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/tokenized_comment_tokens_train.json'
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
tokens_output_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/code_tokens_further.jsonl"
maps_output_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/code_token_maps.jsonl"

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

# load map data
with open(maps_output_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        code_token_maps.append(json_obj['code_token_map'])

# load tokenize id data
with open('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/tokenized_id_train.json', 'r') as f:
    tokenized_id_data = json.load(f)
print("len(tokenized_id_data)", len(tokenized_id_data))     


# Process all training data and save results
print("Processing training data with unique symbols/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling.")

# Save processed docstring tokens to file
docstring_tokens_unique_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/docstring_tokens_unique.json"
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
code_tokens_unique_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/code_tokens_unique.json"
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


system_prompt = "You are an expert at aligning tokens between comments and code. You can accurately identify the similarities and differences between tokens, and you are highly skilled at matching tokens based on their semantics and functionality. You are given input data consisting of comment tokens and code tokens, and your task is to align them by identifying concepts in the comments and matching them to corresponding code tokens. Use the example cases below and output your results in the specified format."

# Load representative examples to get indices for labeling
representative_examples_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/representative_examples.json"
with open(representative_examples_file, 'r') as f:
    representative_examples = json.load(f)

# Extract indices from representative examples
auto_label_indices = [example['index'] for example in representative_examples]
print(f"Number of examples to label: {len(auto_label_indices)}")

# Check if sorted_labelling_sample_api.jsonl exists and load already processed indices
processed_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api.jsonl"
processed_indices = set()
if os.path.exists(processed_file):
    with open(processed_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            processed_indices.add(entry['idx'])
    
    # Filter out already processed indices from auto_label_indices
    auto_label_indices = [idx for idx in auto_label_indices if idx not in processed_indices]
    print(f"Number of remaining examples to label after filtering processed ones: {len(auto_label_indices)}")



# auto labelling 
for i in range(len(auto_label_indices)):
# for auto_label_ind in range(10):
    # Get student and teacher indices from the current pair
    student_idx = auto_label_indices[i]

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

    promt_str = system_prompt + student_prompt

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))

    # 打开日志文件
    with open('./auto_labelling.log', 'a') as log_file:
        log_file.write(f"\n=== Processing auto_label_ind: {i} ===\n")
        log_file.write(f"Student idx: {student_idx}\n")

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
        log_file.write(f"Student index: {student_idx}\n")

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
            
            # Process code tokens with new matching logic
            log_file.write("\nProcessing code tokens:\n")
            for code_token in alignment["code_token"]:
                # Find the unique token in code_tokens_further_data_unique
                try:
                    token_idx = code_tokens_further_data_unique[student_idx].index(code_token)
                    log_file.write(f"Found token {code_token} at index {token_idx}\n")
                    
                    # Find which original token this was derived from using code_token_maps
                    for orig_idx, derived_indices in code_token_maps[student_idx].items():
                        if token_idx in derived_indices:
                            # Convert orig_idx from string to integer before using it
                            orig_idx_int = int(orig_idx)
                            log_file.write(f"Original key {orig_idx} was converted to integer {orig_idx_int}\n")
                            # Get the tokenized range for this original token
                            tokenize_range = tokenized_id_data[student_idx][1:][orig_idx_int]
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
                                    log_file.write(f"No individual matches found - matching available indices {available_indices} for combined token {combined_token} to {clean_token}\n")
                            break
                            
                except ValueError:
                    log_file.write(f"Could not find token {code_token} in unique tokens\n")
                    continue
                    
            log_file.write(f"Final code matches: {code_matches}\n")
            
            # Convert indices to intervals
            comment_intervals = convert_to_intervals(sorted(comment_matches))
            code_intervals = convert_to_intervals(sorted(code_matches))
            log_file.write(f"\nConverted to intervals:\nComment intervals: {comment_intervals}\nCode intervals: {code_intervals}\n")
            
            if len(comment_intervals) > 0 and len(code_intervals) > 0:
                final_results.append([comment_intervals, code_intervals])
                log_file.write(f"Added to final results: {[comment_intervals, code_intervals]}\n")

        if not final_results:
            log_file.write(f"No final results for auto_label_ind: {i}\n")
            continue

        log_file.write(f"Final results: {final_results}\n")

        new_entry = {
            "idx": int(student_idx),
            "match": final_results
        }

        file_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/sorted_labelling_sample_api.jsonl"
        with open(file_path, 'a') as file:
            file.write(json.dumps(new_entry) + '\n')
            log_file.write("Successfully wrote new entry to JSONL file\n")