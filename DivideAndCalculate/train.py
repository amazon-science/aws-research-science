from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
from rewards import calculate_reward_strict, strict_format_reward_func, xmlcount_reward_func, tool_usage_reward, validate_math_expression_reward, numerical_reward_func
from dataset import load_and_combine_datasets
import os

# Replace the last line with this
hf_token = os.getenv("HUGGING_FACE_TOKEN")


max_seq_length = 1024 # Can increase for longer think traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-Math-7B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory

)

template = """{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}
    {%- else %}
        {{- \'<|im_start|>system\\n{system_message}<|im_end|>\\n\' }}
    {%- endif %}\n{%- endif %}\n{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}
    {%- elif message.role == "assistant" %}
        {{- \'<|im_start|>\' + message.role }}
        {%- if message.content %}
            {{- \'\\n\' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- \'\\n<tool_call>\\n{"name": "\' }}
            {{- tool_call.name }}
            {{- \'", "arguments": \' }}
            {{- tool_call.arguments | tojson }}
            {{- \'}\\n</tool_call>\' }}
        {%- endfor %}
        {{- \'<|im_end|>\\n\' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}            {{- \'<|im_start|>user\' }}
        {%- endif %}
        {{- \'\\n<tool_response>\\n\' }}
        {{- message.content }}
        {{- \'\\n</tool_response>\' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- \'<|im_end|>\\n\' }}
        {%- endif %}
    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}
    {{- \'<|im_start|>assistant\\n\' }}
{%- endif %}"""

tokenizer.chat_template = template

# from unsloth.tokenizer_utils import _load_correct_tokenizer
# tokenizer = _load_correct_tokenizer("unsloth/Qwen2.5-Math-7B")

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


################## Transformers library #########################


# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model
# from transformers import BitsAndBytesConfig
# import torch

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# # Load model with 4-bit quantization
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-3B-Instruct",
#     quantization_config=quantization_config,
#     device_map="auto"
# )

# # Configure LoRA
# lora_config = LoraConfig(
#     r=lora_rank,  # Same rank as your original code
#     lora_alpha=lora_rank,  # Same alpha as your original code
#     # target_modules=[
#     #     "q_proj", "k_proj", "v_proj", "o_proj",
#     #     "gate_proj", "up_proj", "down_proj",
#     # ],
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # Apply LoRA
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# # Enable gradient checkpointing if needed
# if hasattr(model, "enable_gradient_checkpointing"):
#     model.enable_gradient_checkpointing()

# # Set maximum sequence length
# model.config.max_sequence_length = max_seq_length







from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 4000,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 7,473,
    # save_steps = 250,
    max_grad_norm = 0.1,
    report_to = ["wandb"], # Can use Weights & Biases
    output_dir = "outputs",
)

dataset = load_and_combine_datasets()

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [calculate_reward_strict, strict_format_reward_func, xmlcount_reward_func, tool_usage_reward, validate_math_expression_reward, numerical_reward_func],
    args = training_args,
    train_dataset = dataset
)
trainer.train()



model.save_lora("grpo_saved_lora")
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged("Satyach/Qwen2.5-Math-7B-Instruct-grpo_token-v2", tokenizer, save_method = "merged_16bit", token = hf_token)