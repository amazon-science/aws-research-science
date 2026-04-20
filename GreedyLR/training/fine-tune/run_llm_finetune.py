import os
import gc
import sys
import time
from datetime import datetime
import argparse
import shutil
import torch
from accelerate import Accelerator
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model, 
    AutoPeftModelForCausalLM
)
from random import randrange
from random import randint
from datasets import load_dataset
from random import randint
from itertools import chain
from functools import partial
import sagemaker
from sagemaker.experiments.run import Run
from smexperiments_callback import SageMakerExperimentsCallback

RANDOM_SEED = 42
transformers.set_seed(RANDOM_SEED)


print("transformers", transformers.__version__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Original
def parse_opt_args(opt_string):
    params = opt_string.split('|')
    
    params_dict = {}
    for param in params:
        _key, values = param.split('$')
        params_dict[_key] = {v.split(':')[0]: eval(v.split(':')[1]) for v in values.split(',')}
        
    return params_dict

# Original
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Greedy LR Testing")

    # dataset
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--test", type=str)
    
    # train args
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--logging_dir", type=str)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--bf16", type=bool)
    parser.add_argument("--save_strategy", type=str)
    
    # peft
    parser.add_argument("--peft_rvalue", type=int)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--scheduler_args", type=str)
    

    # Parse the arguments
    args = parser.parse_args()

    return args


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    
def format_simple_cot(sample):
    instruction = f"### Instruction\n{sample['source']}"
    answer = f"### Answer\n{sample['target']}" 
    response = f"### Rationale\n{sample['rationale']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, answer, response] if i is not None])
    return prompt

def format_xP3mt(sample):
    instruction = f"### Instruction\n{sample['inputs']}"
    answer = f"### Answer\n{sample['targets']}" 
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, answer] if i is not None])
    return prompt


def run(_args):
    
    accelerator = Accelerator()
    
    # model_id = "tiiuae/falcon-7b"
    model_id = "microsoft/phi-2"
    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # The code is provided by the model authors in the repo.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        # quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=int(_args.peft_rvalue),
        lora_alpha=int(_args.peft_rvalue * 2),
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.08,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    # Specifically for EleutherAI/lambada_openai / xP3mt
#     train_dataset = load_dataset(_args.dataset_name, 'code', split=_args.train)
#     validation_dataset = load_dataset(_args.dataset_name, 'code', split=_args.valid)
#     test_dataset = load_dataset(_args.dataset_name, 'code', split=_args.test)
    
    # General HF dataset; e.g. w601sxs/simpleCoT
    train_dataset = load_dataset(_args.dataset_name, split=_args.train)
    validation_dataset = load_dataset(_args.dataset_name, split=_args.valid)
    test_dataset = load_dataset(_args.dataset_name, split=_args.test)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # template dataset to add prompt to each sample
    def template_dataset(sample):
        # NOTE: commented out cell below for lambada/xP3mt dataset
        sample["text"] = f"{format_simple_cot(sample)}{tokenizer.eos_token}"
#         sample["text"] = f"{format_xP3mt(sample)}{tokenizer.eos_token}"
        return sample

    # apply prompt template per sample
    # NOTE: comment out cells below for lambada dataset
    # train
    train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))
    # validation
    validation_dataset = validation_dataset.map(template_dataset, remove_columns=list(validation_dataset.features))
    # test
    test_dataset = test_dataset.map(template_dataset, remove_columns=list(test_dataset.features))

    # print random sample
    print(validation_dataset[randint(0, len(validation_dataset))]["text"])
    
    # empty list to save remainder from batches to use in next batch
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    def chunk(sample, remainder, chunk_length=2048):
        
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # get total number of tokens for batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # get max number of chunks for batch
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # add remainder to global variable for next batch
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # prepare labels
        result["labels"] = result["input_ids"].copy()
        return result
    
    # training
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(train_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=2048),
        batched=True,
    )

    # validation
    lm_valid_dataset = validation_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(validation_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=2048),
        batched=True,
    )

    # validation
    lm_test_dataset = test_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(test_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=2048),
        batched=True,
    )

    # Print total number of samples
    print(f"Train Length : {len(lm_train_dataset)} || Val Length : {len(lm_valid_dataset)} || Test Length : {len(lm_test_dataset)}")
    
    chosen_scheduler = _args.scheduler
    
    run_name = f"run-{_args.scheduler}-r{_args.peft_rvalue}a{int(2*_args.peft_rvalue)}-{datetime.now().strftime('%Y%m%d%H%M')}"
    exp_name = f"GreedyLr-seed{RANDOM_SEED}-{datetime.now().strftime('%Y%m%d%H%M')}"
    
    with Run(
        experiment_name=exp_name, 
        run_name=run_name, 
        sagemaker_session=sagemaker.Session()
    ) as run:
    
        logging_dir = f"./logs/{exp_name}/{run_name}/tensorboard"
        output_dir = f"./logs/{exp_name}/{run_name}/output"
        temp_dir = f"./logs/{exp_name}/{run_name}/peft-model-dir"
        # model_dir = f"./model-outputs/{lr_schedulers[chosen_scheduler]}/fine-tuned-model-dir"

        run.log_parameters(
            {
                "exp_current_scheduler": chosen_scheduler,
                "exp_logging_dir": logging_dir,
                "exp_output_dir": output_dir,
                "exp_peft_dir": temp_dir
            }
        ) 
        
        sch_dict = parse_opt_args(_args.scheduler_args)
        print(_args.scheduler, "=======>", sch_dict[_args.scheduler])
        
      
        if _args.scheduler == "greedy":
            
            # print(_args.scheduler, "GREEDY =======>", sch_dict[_args.scheduler])
            train_args = transformers.TrainingArguments(
                per_device_train_batch_size=_args.per_device_train_batch_size,
                per_device_eval_batch_size=_args.per_device_eval_batch_size,
                logging_dir=_args.logging_dir,
                logging_steps=2,
                num_train_epochs=_args.num_train_epochs,
                learning_rate=_args.learning_rate,
                bf16=False,
                save_strategy=_args.save_strategy,
                output_dir=output_dir,
                report_to="tensorboard",
                lr_scheduler_type=_args.scheduler,
                # greedy
                min_lr=sch_dict[_args.scheduler]['min_lr'],
                smooth=sch_dict[_args.scheduler]['smooth'],
                factor=sch_dict[_args.scheduler]['factor']
            )
            
        elif _args.scheduler == "cosine":
            # print(_args.scheduler, "COSINE=======>", sch_dict[_args.scheduler])
            
            train_args = transformers.TrainingArguments(
                per_device_train_batch_size=_args.per_device_train_batch_size,
                per_device_eval_batch_size=_args.per_device_eval_batch_size,
                logging_dir=_args.logging_dir,
                logging_steps=2,
                num_train_epochs=_args.num_train_epochs,
                learning_rate=_args.learning_rate,
                bf16=False,
                save_strategy=_args.save_strategy,
                output_dir=output_dir,
                report_to="tensorboard",
                lr_scheduler_type=_args.scheduler,
                # cosine
                warmup_ratio=sch_dict[_args.scheduler]['warmup_ratio']
                # eta_min=sch_dict[_args.scheduler]['eta_min']
            )
            
        elif _args.scheduler == "constant":
            # print(_args.scheduler, "COSINE=======>", sch_dict[_args.scheduler])
            
            train_args = transformers.TrainingArguments(
                per_device_train_batch_size=_args.per_device_train_batch_size,
                per_device_eval_batch_size=_args.per_device_eval_batch_size,
                logging_dir=_args.logging_dir,
                logging_steps=2,
                num_train_epochs=_args.num_train_epochs,
                learning_rate=_args.learning_rate,
                bf16=False,
                save_strategy=_args.save_strategy,
                output_dir=output_dir,
                report_to="tensorboard",
                lr_scheduler_type=_args.scheduler,
                # constant
                warmup_ratio=sch_dict[_args.scheduler]['warmup_ratio'],
                weight_decay=0.1
                # eta_min=sch_dict[_args.scheduler]['eta_min']
            )

            
        trainer = accelerator.prepare(transformers.Trainer(
            model=model,
            train_dataset=lm_train_dataset,
            eval_dataset=lm_valid_dataset,
            args=train_args,
            callbacks=[SageMakerExperimentsCallback(region="us-east-1")],
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer,
                mlm=False
            )
        ))
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

        trainer.train()

        trainer.model.save_pretrained(
            temp_dir, 
            safe_serialization=False
        )
        
        print('moving model to CPU')
        model.to('cpu')

        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        print("sleep...")
        time.sleep(2)
        print("continue!")
    
    
    
    #     model = AutoPeftModelForCausalLM.from_pretrained(
    #         temp_dir,
    #         # low_cpu_mem_usage=True,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #     )
    #     model = model.merge_and_unload()

    #     model.save_pretrained(
    #         model_dir, 
    #         safe_serialization=True, 
    #         max_shard_size="9GB"
    #     )
    #     tokenizer.save_pretrained(
    #         save_directory=model_dir, 
    #         from_pt=True
    #     )

    #     shutil.rmtree(temp_dir)
    
    return 0



if __name__ == "__main__":
    args = parse_args()
    
    print("************************************")
    print(f"Running training with {args}")
    print("************************************")
    print("\n")
    
    run(args)