#!/bin/bash

dataset_name=$(jq -r '.dataset.name' config.json)
dataset_train=$(jq -r '.dataset.train' config.json)
dataset_valid=$(jq -r '.dataset.valid' config.json)
dataset_test=$(jq -r '.dataset.test' config.json)

per_device_train_batch_size=$(jq '.base_train_args.per_device_train_batch_size' config.json)
per_device_eval_batch_size=$(jq '.base_train_args.per_device_eval_batch_size' config.json)
logging_steps=$(jq '.base_train_args.logging_steps' config.json)
num_train_epochs=$(jq '.base_train_args.num_train_epochs' config.json)
learning_rate=$(jq '.base_train_args.learning_rate' config.json)
bf16=$(jq '.base_train_args.bf16' config.json)
save_strategy=$(jq -r '.base_train_args.save_strategy' config.json)
output_dir=$(jq -r '.base_train_args.output_dir' config.json)
report_to=$(jq -r '.base_train_args.report_to' config.json)

peft_r_alpha_ranges=$(jq -r '.peft_ranges.r_alpha_values[]' config.json)
schedulers=$(jq -r '.schedulers' config.json)

# Echo print all the variables
echo "Dataset Name: $dataset_name"
echo "Dataset Train: $dataset_train"
echo "Dataset Valid: $dataset_valid"
echo "Dataset Test: $dataset_test"

echo "Per Device Train Batch Size: $per_device_train_batch_size"
echo "Per Device Eval Batch Size: $per_device_eval_batch_size"

echo "Logging Steps: $logging_steps"
echo "Num Train Epochs: $num_train_epochs"
echo "Learning Rate: $learning_rate"
echo "BF16: $bf16"
echo "Save Strategy: $save_strategy"
echo "Report To: $report_to"

echo "R Alpha Ranges: $peft_r_alpha_ranges"
echo "Schedulers: $schedulers"

for sch in "greedy"; do
    for value in $peft_r_alpha_ranges; do
      /home/ec2-user/anaconda3/envs/py310-greedy/bin/python3 /home/ec2-user/SageMaker/transformers/examples/pytorch/greedy-lr/run_llm_finetune.py --dataset_name $dataset_name --train $dataset_train --valid $dataset_valid --test $dataset_test \
      --per_device_train_batch_size $per_device_train_batch_size --per_device_eval_batch_size $per_device_eval_batch_size \
      --logging_steps $logging_steps --num_train_epochs $num_train_epochs --learning_rate $learning_rate --bf16 $bf16 --save_strategy $save_strategy \
      --peft_rvalue $value --scheduler $sch --scheduler_args $schedulers
      sleep 5
    done
done
