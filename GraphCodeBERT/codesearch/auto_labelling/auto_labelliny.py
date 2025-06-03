import json
import os
import re
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer
from hanlp_restful import HanLPClient # pip install hanlp_restful
from openai import OpenAI
import httpx
import time
import random
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from urllib.error import HTTPError

os.environ["OPENAI_API_KEY"] = "sk-KeHDmzqGVMtMycIwE70f74Cc57E34c899d39E041177f44F2"
os.environ["OPENAI_BASE_URL"] = "https://api.vveai.com/v1"

if __name__ == '__main__':
    existing_data = []
    start_index = 0
    end_index = 2000
    output_path = f"/home/zicong/Project/CodeSearch/auto_label/filter_low_quality_data/lack_func_call_0.9_labeled_05300500.jsonl"

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        existing_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"行 {line_num} 的JSON解析错误: {e}")
                        with open("json_error_lines.txt", "a", encoding="utf-8") as err_file:
                            err_file.write(f"行 {line_num}: {line}\n")
                        assert False, f"行 {line_num} 的JSON解析错误: {e}"

    print(f"已读取 {len(existing_data)} 条已处理的数据")
    start_index = start_index + len(existing_data)
    print(f"从第 {start_index} 条数据开始处理")

    train_file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/train.jsonl'
    train_data = []
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))

    teacher_input_path = "/home/zicong/Project/CodeSearch/CodeBERT/dataset/python/train.jsonl"
    teacher_data_full = {}  

    try:
        with open(teacher_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().rstrip(',')
                if line:
                    json_obj = json.loads(line)
                    teacher_data_full[json_obj['idx']] = json_obj
        print(f"Loaded {len(teacher_data_full)} teacher samples with complete data")
    except FileNotFoundError:
        print(f"Warning: {teacher_input_path} not found, will use fallback method")
        teacher_data_full = {}

    # 加载student-teacher配对
    student_teacher_pairs_file = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/student_teachers_pairs_more_reference_loose.jsonl"
    student_teacher_pairs = []
    if os.path.exists(student_teacher_pairs_file):
        with open(student_teacher_pairs_file, 'r') as f:
            for line in f:
                student_teacher_pairs.append(json.loads(line))
        print(f"Loaded {len(student_teacher_pairs)} student-teacher pairs")

    hanlp_auth = "ODQyMUBiYnMuaGFubHAuY29tOm1oRngwZUNjSmIyTWdUVHo="
    HanLP = HanLPClient('https://www.hanlp.com/api', auth=hanlp_auth, language='en')

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    data = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    
    with open(train_file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_index:
                continue
            if line_idx > end_index:
                break
                
            line = line.strip()
            if not line:
                continue
                
            obj = json.loads(line)
            
            print(f"正在处理第 {line_idx+1} 条数据（跳过前 {start_index} 条）")

            if len(obj["docstring"]) > 4700:
                obj["docstring"] = obj["docstring"][:4700]

            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    wait_time = 0.9 + random.uniform(0.1,0.2)
                    print(f"处理第 {line_idx+1} 条数据，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                    
                    doc = HanLP.parse(obj["docstring"], tasks='dep')
                    obj["docstring_tokens"] = doc["tok"][0]
                    obj["docstring_dep"] = doc.to_pretty()[0]
                    break
                except HTTPError as e:
                    retry_count += 1
                    if "429" in str(e) and retry_count < max_retries:
                        wait_time = 10 + random.uniform(2, 5)
                        print(f"遇到速率限制，第 {retry_count} 次重试，等待 {wait_time:.2f} 秒...")
                        time.sleep(wait_time)
                    else:
                        print(f"处理失败: {e}")
                        if retry_count >= max_retries:
                            print(f"已达到最大重试次数，跳过该条数据")
                            continue

            comment_tokens = obj["docstring_tokens"]
            comment_dependency = obj["docstring_dep"]
            code_tokens = obj["code_tokens"]

            current_student_idx = line_idx
            
            matching_pair = None
            for pair in student_teacher_pairs:
                if pair['student_idx'] == current_student_idx:
                    matching_pair = pair
                    break
            
            dep_prompt = [
                {
                    "role": "developer",
                    "content": """
                        You are an code-comment alignment extractor. 
                        Inputs:
                          (1) comment_tokens: List[str]
                          (2) comment dependency graph: str
                          (2) code_tokens:   List[str]
                        Outputs (JSON only):
                          {
                            "COMMENT_CONCEPTS": List[{"Concept N": List[str]}],
                            "ALIGNMENT_MAP":   List[{"Concept N": List[str]}]
                          }
                        # Chain-of-thought steps: 
                            Step 1: pick the `root` comment tokens, by analyzing the syntactic and semantic dependencies and output `COMMENT_CONCEPTS`. Ensure that each comment token occurrence is only used once across all concepts.
                            Step 2: with step 1's output, find the code tokens that `implement` the corresponding comment concepts, and output `ALIGNMENT_MAP`. When assigning code tokens:
                                - Check if the code token occurrence has already been assigned to another concept.
                                - If it has, skip it to ensure one code token occurrence is only assigned once.
                                - If it hasn't, assign it to the current concept.
                            step 3: ensure that the alignment adheres to the constraints listed below.
                        # Constraints:  
                           - One code token occurrence can only be assigned to one concept
                           - One comment token occurrence can only be assigned to one concept
                           - Some code tokens and comment tokens may remain unaligned
                           - Preserve original token formatting
                           - Consider both explicit and implicit relationships between code tokens and comment concepts
                           - Ensure that important code tokens, especially those involving function calls, API usages, and key operations, are included in the alignment
                           - Output must be valid JSON, no explanations or extra keys
                    """
                }
            ]
            
            teacher_examples_added = False
            if matching_pair:
                teachers = matching_pair['teachers'][:3]
                
                for teacher_info in teachers:
                    teacher_idx = teacher_info['teacher_idx']
                    
                    if teacher_idx in teacher_data_full:
                        teacher_full_data = teacher_data_full[teacher_idx]
                        teacher_comment_tokens = teacher_full_data.get('docstring_tokens', [])
                        teacher_comment_dependency = teacher_full_data.get('docstring_dep', "")
                        teacher_code_tokens = teacher_full_data.get('code_tokens', [])
                        teacher_response = teacher_full_data.get('response', '{}')
                        
                        if teacher_comment_tokens and teacher_code_tokens and teacher_response != '{}':
                            dep_prompt.extend([
                                {
                                    "role": "user",
                                    "content": f"""(1) comment tokens: {teacher_comment_tokens},
                                        (2) comment tokens dependency graph: "{teacher_comment_dependency}",
                                        (3) code tokens: {teacher_code_tokens}
                                        """
                                },
                                {
                                    "role": "assistant",
                                    "content": teacher_response
                                }
                            ])
                            teacher_examples_added = True
                            break
            
            if not teacher_examples_added:
                dep_prompt.extend([
                    {
                        "role": "user",
                        "content": """(1) comment tokens: ["Downloads", "Dailymotion", "videos", "by", "URL", "."],
                            (2) comment tokens dependency graph: "Dep Tree	Token      	Relation
                                                        ────────	───────────	────────
                                                        ┌┬┬─────	Downloads  	root
                                                        │││  ┌─►	Dailymotion	compound
                                                        ││└─►└──	videos     	obj
                                                        ││   ┌─►	by         	case
                                                        │└──►└──	URL        	obl
                                                        └──────►	.          	punct",
                            (3) code tokens: ['def', 'dailymotion', '_', 'download', '(', 'url', ',', 'output', '_', 'dir', '=', "'", '.', "'", ',', 'merge', '=', 'True', ',', 'info', '_', 'only', '=', 'False', ',', '*', '*', 'kwargs', ')', ':', 'html', '=', 'get', '_', 'content', '(', 'rebuilt', '_', 'url', '(', 'url', ')', ')', 'info', '=', 'json', '.', 'loads', '(', 'match1', '(', 'html', ',', 'r', "'", 'qualities', '"', ':', '(', '{', '.', '+', '?', '}', ')', ',', '"', "'", ')', ')', 'title', '=', 'match1', '(', 'html', ',', 'r', "'", '"', 'video', '_', 'title', '"', '\\', 's', '*', ':', '\\', 's', '*', '"', '(', '[', '^', '"', ']', '+', ')', '"', "'", ')', 'or', 'match1', '(', 'html', ',', 'r', "'", '"', 'title', '"', '\\', 's', '*', ':', '\\', 's', '*', '"', '(', '[', '^', '"', ']', '+', ')', '"', "'", ')', 'title', '=', 'unicodize', '(', 'title', ')', 'for', 'quality', 'in', '[', "'", '1080', "'", ',', "'", '720', "'", ',', "'", '480', "'", ',', "'", '380', "'", ',', "'", '240', "'", ',', "'", '144', "'", ',', "'", 'auto', "'", ']', ':', 'try', ':', 'real', '_', 'url', '=', 'info', '[', 'quality', ']', '[', '1', ']', '[', '"', 'url', '"', ']', 'if', 'real', '_', 'url', ':', 'break', 'except', 'KeyError', ':', 'pass', 'mime', ',', 'ext', ',', 'size', '=', 'url', '_', 'info', '(', 'real', '_', 'url', ')', 'print', '_', 'info', '(', 'site', '_', 'info', ',', 'title', ',', 'mime', ',', 'size', ')', 'if', 'not', 'info', '_', 'only', ':', 'download', '_', 'urls', '(', '[', 'real', '_', 'url', ']', ',', 'title', ',', 'ext', ',', 'size', ',', 'output', '_', 'dir', '=', 'output', '_', 'dir', ',', 'merge', '=', 'merge', ')']
                            """
                    },
                    {
                        "role": "assistant",
                        "content": """{"COMMENT_CONCEPTS": [
                                                        {'Concept 1': ['Downloads']},
                                                        {'Concept 2': ['videos']},
                                                        {'Concept 3': ['URL']}],
                                       "ALIGNMENT_MAP": [
                                                        {"Concept 1": ["download", "download", "get"]},
                                                        {"Concept 2": ["video", "title"]},
                                                        {"Concept 3": ["url", "rebuilt", "url", "real", "url", "urls"]}]
                                       }"""
                    }
                ])
            
            dep_prompt.append({
                "role": "user",
                "content": f"""(1) comment tokens: {comment_tokens},
                    (2) comment tokens dependency graph: "{comment_dependency}",
                    (3) code tokens: {code_tokens}
                    """
            })

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=dep_prompt
                )
                response = response.choices[0].message.content
                print(response)
                
                # test the vaildity of the response
                try:
                    json_response = json.loads(response)
                    obj["response"] = response
                except json.JSONDecodeError:
                    obj["response"] = "{}"
            except Exception as e:
                print(f"OpenAI api call failure: {e}")
                obj["response"] = "{}"
            
            data.append(obj)
            
            if len(data) % 5 == 0:
                with open(output_path, "a", encoding="utf-8") as fw:
                    for entry in data[-5:]:
                        fw.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # save all remaining data
    if data and len(data) % 5 != 0:
        remaining_count = len(data) % 5
        with open(output_path, "a", encoding="utf-8") as fw:
            for entry in data[-remaining_count:]:
                fw.write(json.dumps(entry, ensure_ascii=False) + "\n")