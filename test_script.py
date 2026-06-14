import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct')

def format_data(data):
    instruction = data.get('prompt', '') or data.get('instruction', '')
    input_code = data.get('input', '')
    completion = data.get('completion', '') or data.get('output', '')
    
    full_prompt = f"{instruction}\n\n{input_code}" if input_code else instruction
    
    messages = [
        {'role': 'user', 'content': full_prompt},
        {'role': 'assistant', 'content': completion}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

found_vulnerable = False
found_safe = False

with open('train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if found_vulnerable and found_safe: break
        
        data = json.loads(line)
        output = data.get('completion', '') or data.get('output', '') or ''
        
        output_lower = output.lower()
        is_safe = not output_lower.startswith('to fix') and not output_lower.startswith('to prevent') and not output_lower.startswith('here is')
        
        if is_safe and not found_safe:
            print('\n================== SAFE EXAMPLE ==================')
            print(format_data(data))
            found_safe = True
            
        elif not is_safe and not found_vulnerable:
            print('\n================== VULNERABLE EXAMPLE ==================')
            print(format_data(data))
            found_vulnerable = True
