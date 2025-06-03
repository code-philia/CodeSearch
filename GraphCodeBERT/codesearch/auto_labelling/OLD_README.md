# Auto Labelling Environment Setup

## Step 1: Create Conda Environment

Create a new conda environment with Python 3.10:
```
conda create --name autolabel python=3.10
```

Then activate the environment:
```
conda activate autolabel
```
Install the required packages:
```
pip install -r requirements.txt
```

## Step 2: Run autolabel scripts

Before running the scripts, make sure to update the file paths in the code. Replace all instances of:
```
/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/auto_labelling/
```
with:
```
./auto_labelling/ (current directory)
```

Run the auto labelling scripts in order:
```
python auto_labelling_format_unique_sample.py
```
Note: By default, the script will label 5 samples each time. You can modify the number of samples to be labeled by changing line 267:
```
for auto_label_ind in range(5):
```

Then you will get a new file `auto_label_unique.jsonl` in the current directory.
Then you can run the labelling_observation.ipynb script to visulize the labelling results using the same conda environment.

Note: You can adjust the indice variable to visulize different samples.