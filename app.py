import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")

print("Initializing AI Application Security Engineer UI...")
print("Loading Massive 32B Base Model into 192GB VRAM...")

# 1. Load the native 16-bit model (no quantization bugs!)
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Applying Custom Java Vulnerability Weights...")
model = PeftModel.from_pretrained(base_model, "./large-java-vuln-adapter")
model.config.use_cache = True
model.eval()

print("Model Loaded Successfully! Starting Web Server...")

def analyze_code(java_code):
    if not java_code.strip():
        return "Please paste some Java code to analyze."
        
    # Using the EXACT prompt the model was trained on!
    prompt = f"Analyze the following Java code. If a vulnerability exists, provide the fixed code. If it is safe, output the original code.\n\n{java_code}"
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.1,     
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

# 2. Build the Beautiful Web UI
demo = gr.Interface(
    fn=analyze_code,
    inputs=gr.Code(language="java", label="Input Java Code", lines=15),
    outputs=gr.Markdown(label="AI Security Analysis & Patch"),
    title="🛡️ AI Application Security Engineer",
    description="Paste any Java code below. This 32-Billion parameter model (fine-tuned on an AMD MI300X) will automatically detect vulnerabilities, explain the threat, and rewrite the code safely.",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo").set(
        body_background_fill="*background_fill_secondary"
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    # Runs the server locally. In an AMD portal, port 7860 is usually exposed or port-forwardable.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
