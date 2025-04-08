lang=python

CUDA_VISIBLE_DEVICES=2 python run2.py \
--output_dir /home/yiming/cophi/training_dynamic/graphcodebert \
--config_name /home/yiming/cophi/projects/graphcodebert-base \
--model_name_or_path /home/yiming/cophi/projects/graphcodebert-base \
--tokenizer_name /home/yiming/cophi/projects/graphcodebert-base \
--lang=$lang \
--do_train \
--train_data_file=dataset/$lang/train.jsonl \
--eval_data_file=dataset/$lang/valid.jsonl \
--test_data_file=dataset/$lang/test.jsonl \
--codebase_file=dataset/$lang/codebase.jsonl \
--num_train_epochs 20 \
--code_length 256 \
--data_flow_length 64 \
--nl_length 128 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--seed 123456 \
2>&1| tee train2.log