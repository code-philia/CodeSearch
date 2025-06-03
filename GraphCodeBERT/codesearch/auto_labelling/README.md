# Code-Comment Alignment Auto-Labeling System

## Usage

### Requirements

```bash
pip install transformers torch numpy hanlp_restful openai httpx jinja2
```

### Required Input Files

#### 1. Training Data File (`train.jsonl`)

Contains basic training dataset information:
```json
{
    "repo": "smdabdoub/phylotoast", 
    "path": "phylotoast/util.py", 
    "func_name": "split_phylogeny", 
    "original_string": "def split_phylogeny(p, level=\"s\"):\n    \"\"\"\n    Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.\n    \"\"\"\n    level = level+\"__\"\n    result = p.split(level)\n    return result[0]+level+result[1].split(\";\")[0]", 
    "language": "python", 
    "code": "def split_phylogeny(p, level=\"s\"):\n    \"\"\"\n    Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.\n    \"\"\"\n    level = level+\"__\"\n    result = p.split(level)\n    return result[0]+level+result[1].split(\";\")[0]", 
    "code_tokens": ["def", "split_phylogeny", "(", "p", ",", "level", "=", "\"s\"", ")", ":", "level", "=", "level", "+", "\"__\"", "result", "=", "p", ".", "split", "(", "level", ")", "return", "result", "[", "0", "]", "+", "level", "+", "result", "[", "1", "]", ".", "split", "(", "\";\"", ")", "[", "0", "]"], 
    "docstring": "Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.", 
    "docstring_tokens": ["Return", "either", "the", "full", "or", "truncated", "version", "of", "a", "QIIME", "-", "formatted", "taxonomy", "string", "."], 
    "sha": "0b74ef171e6a84761710548501dfac71285a58a3", 
    "url": "https://github.com/smdabdoub/phylotoast/blob/0b74ef171e6a84761710548501dfac71285a58a3/phylotoast/util.py#L159-L177", 
    "partition": "train", 
    "idx": 0
}
```

#### 2. Teacher Samples File (`sorted_labelling_sample_api.jsonl`)

Contains high-quality manually labeled teacher samples:
```json
{
    "idx": 123,
    "docstring": "Downloads Dailymotion videos by URL.",
    "docstring_tokens": ["Downloads", "Dailymotion", "videos", "by", "URL", "."],
    "docstring_dep": "Dep Tree\tToken\tRelation\n────────\t───────────\t────────\n┌┬┬─────\tDownloads\troot\n...",
    "code_tokens": ["def", "dailymotion", "_", "download", "..."],
    "response": "{\"COMMENT_CONCEPTS\": [{\"Concept 1\": [\"Downloads\"]}, {\"Concept 2\": [\"videos\"]}, {\"Concept 3\": [\"URL\"]}], \"ALIGNMENT_MAP\": [{\"Concept 1\": [\"download\", \"download\", \"get\"]}, {\"Concept 2\": [\"video\", \"title\"]}, {\"Concept 3\": [\"url\", \"rebuilt\", \"url\", \"real\", \"url\", \"urls\"]}]}"
}
```

#### 3. Student-Teacher Pairs File (`student_teachers_pairs_more_reference_loose.jsonl`)

Defines matching relationships between student and teacher samples:
```json
{
    "student_idx": 5000,
    "teachers": [
        {"teacher_idx": 123, "confidence": 0.85},
        {"teacher_idx": 456, "confidence": 0.78},
        {"teacher_idx": 789, "confidence": 0.72}
    ]
}
```

## Annotation Pipeline

### 1. Data Preprocessing
- **HanLP Dependency Parsing**: Parse comment text to generate syntactic dependency trees
- **Token Extraction**: Extract comment tokens and dependency relationships
- **Teacher Matching**: Find similar teacher samples based on student-teacher pairs

### 2. Prompt Construction Strategy

The system uses a in-context learning approach with the following structure:

#### Base System Prompt
```
You are a code-comment alignment extractor.
Inputs:
  (1) comment_tokens: List[str]
  (2) comment dependency graph: str
  (3) code_tokens: List[str]
Outputs (JSON only):
  {
    "COMMENT_CONCEPTS": List[{"Concept N": List[str]}],
    "ALIGNMENT_MAP": List[{"Concept N": List[str]}]
  }
```

#### Dynamic Teacher Example Selection
The system dynamically selects teacher examples :
   - If `student_teacher_pairs_file` contains matching pairs for current sample
   - Uses up to 3 most similar teacher samples
   - Teacher examples include complete dependency graphs and alignment results

#### Prompt Template Structure
```
[System Role: Developer with task description]
↓
[Teacher Example (User Input)]
↓ 
[Teacher Example (Assistant Response)]
↓
[Current Sample (User Input)]
```

### 3. Chain-of-Thought Reasoning

The prompt guides GPT-4 through a structured reasoning process:

1. **Step 1**: Identify root comment concepts using dependency analysis
2. **Step 2**: Map comment concepts to implementing code tokens
3. **Step 3**: Ensure one-to-one token assignment constraints

### 4. Annotation Workflow

```mermaid
    A[Load Input Sample] --> B[HanLP Dependency Parsing]
    B --> C[Find Teacher Examples]
    C --> D[Construct Prompt with Examples]
    D --> E[GPT API Call]
    E --> F[Save Result]
```

### Run

1. Configure file paths in the script:
```python
train_file_path = "your_train_file.jsonl"
output_path = "your_output_file.jsonl"
teacher_sample_path = "your_teacher_sample_file.jsonl"
student_teacher_pairs_file = "your_student_teacher_pairs_file.jsonl"
hanlp_auth = "your_hanlp_auth_token"
start_index = 0
end_index = 1000
```

2. Run the script:
```bash
python auto_labelliny.py
```

### Notes

- The system supports resume processing from interruption points
- Processes data in batches and saves every 5 samples
- Includes automatic retry mechanism for API calls
- Teacher example selection significantly improves annotation quality over static examples

## Plan

### Data Volume

| PL         | Training |  Annotated   |  Manually Labeled |
| :--------- | :------: | :----: | :----: |
| Python     | 251,820  | 150k | 50/50 |
| PHP        | 241,241  | / | 0/50 |
| Go         | 167,288  | /  |  0/30 |
| Java       | 164,923  | /  | 0/30 |
| JavaScript |  58,025  | /  |  0/12 |
| Ruby       |  24,927  | /  |  0/10 |

**Current Progress:**
- Auto labeling speed: 10k-20k samples per API account per day
- Estimated completion time: 1-2 weeks for all planned annotations

### Research Questions

| RQ | Question | Evaluation Method | Metrics | Dataset |
|:--|:---------|:-----------------|:--------|:--------|
| RQ1 | Overall Retrieval Performance | Compare with baselines | MRR | ID: CodeSearchNet: Python, PHP, Go, Java, JavaScript, Ruby<br>OOD: COSQA+ |
| RQ2 | Alignment & Attention Accuracy | Compare with baselines | Alignment Precision & Recall, Attention Precision & Recall, Retrieval Specific Recall | CodeSearchNet: Python, PHP, Go, Java, JavaScript, Ruby |
| RQ3 | Component Contribution | Remove components | MRR | CodeSearchNet: Python, PHP, Go, Java, JavaScript, Ruby |
| RQ4 | User Acceptance | User study | Acceptance Rate | 10-20 samples |
