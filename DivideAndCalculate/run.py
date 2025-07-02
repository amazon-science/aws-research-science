from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria
from accelerate import Accelerator
import torch
from utils import generate_with_api

from transformers import BitsAndBytesConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria
from accelerate import Accelerator
import torch
import logging
#from bitsandbytes import Bits 

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16
)



# Initialize accelerator
accelerator = Accelerator()
# Load model and tokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",          # Automatic GPU/CPU allocation
    torch_dtype="auto",         # Automatic precision selection
    trust_remote_code=True,     # Required for some custom models
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
    # token="hf_YourAPIKey"       # For gated repositories
)

accelerator
model = accelerator.prepare(model)





# Move inputs to the same device as the model
# inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
system_prompt = """

use a calculator by following this format

<api> [Calculator( The expression to be calculated )] </api>

Here is an example for reference


solve 18 + 12 x 3

<think>
okay, I have to solve this expression "18 + 12 x 3". let me use the calculator to solve this expression

<api> [Calculator((18+12)*3)] </api> -> 54

I got 54 as an answer from the Calculator. So the answer is 54

</think>
<answer>
54
<answer>


"""

question = """
	
For a pair $ A \equal{} (x_1, y_1)$ and $ B \equal{} (x_2, y_2)$ of points on the coordinate plane, let $ d(A,B) \equal{} |x_1 \minus{} x_2| \plus{} |y_1 \minus{} y_2|$. We call a pair $ (A,B)$ of (unordered) points [i]harmonic[/i] if $ 1 < d(A,B) \leq 2$. Determine the maximum number of harmonic pairs among 100 points in the plane.
"""

final_prompt = question + system_prompt
# Generate text
inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
# Move inputs to the same device as the model
# inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

output = generate_with_api(inputs.input_ids, model=model, tokenizer=tokenizer, max_new_tokens=1000)

print(output)

