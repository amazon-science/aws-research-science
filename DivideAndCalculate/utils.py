import re
import numexpr as ne
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def extract_last_calculator_expression(prompt):
    # Pattern to match content between <api> and </api> tags that contain Calculator
    pattern = r'<api>\s*\[Calculator\((.*?)\)\]\s*</api>'
    
    # Find all matches
    matches = re.finditer(pattern, prompt)
    
    # Get the last match (if any)
    last_match = None
    for match in matches:
        last_match = match
    
    if last_match:
        return last_match.group(1)  # Return the captured expression
    return None


def calculate_safe(expression):
    """A safer version using numexpr instead of eval"""
    try:
        # Remove all spaces from the expression
        cleaned_expression = expression.replace(" ", "")
        #replace ^ with ** for exponentiation
        cleaned_expression = cleaned_expression.replace("^", "**")
        result = ne.evaluate(cleaned_expression).item()
        print(result)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def is_last_word_api_tag(text):
    # Method 1: Using strip() and endswith()
    cleaned_text = text.strip()  # Remove leading/trailing whitespace
    return cleaned_text.endswith('</api>')


def generate_with_api(inputs, model, tokenizer, max_new_tokens=1000, max_depth=5, current_depth=0):
    """
    Generate text with API-like calculation capabilities
    
    Args:
        inputs: Tokenized input tensor
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    
    Returns:
        str: Generated text output
    """
    # Check recursion depth
    if current_depth >= max_depth:
        print("Maximum recursion depth reached")
        return tokenizer.decode(inputs[0], skip_special_tokens=True)
        
    try:
        outputs = model.generate(
            inputs,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([
                StopStringCriteria(tokenizer=tokenizer, stop_strings=["</api>", "</answer>"])
            ])
        )
        
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if is_last_word_api_tag(output):
            try:
                expression = extract_last_calculator_expression(output)
                answer = calculate_safe(str(expression))
                reprompt = f"{output} -> {answer}"
                print(reprompt)
                
                inputs = tokenizer(reprompt, return_tensors="pt").to(model.device)
                return generate_with_api(
                    inputs.input_ids,
                    model, tokenizer,
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
                
            except (ValueError, ArithmeticError) as e:
                logger.error(f"Calculation error: {str(e)}")
                return output
                
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return tokenizer.decode(inputs[0], skip_special_tokens=True)
        
    return output
