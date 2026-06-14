import json

safe_count = 0
vulnerable_count = 0
first_16000 = 0

with open('train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 16000:
            break
        first_16000 += 1
        data = json.loads(line)
        input_code = data.get('input', '').strip()
        output = data.get('completion', '') or data.get('output', '') or ''
        output = output.strip()

        # Safe = input and output are identical
        if input_code == output:
            safe_count += 1
        else:
            vulnerable_count += 1

print(f"First 16,000 examples (what the model actually sees in 500 steps):")
print(f"  Safe examples:       {safe_count} ({safe_count/first_16000*100:.1f}%)")
print(f"  Vulnerable examples: {vulnerable_count} ({vulnerable_count/first_16000*100:.1f}%)")
print(f"\nThis is {'BALANCED' if abs(safe_count - vulnerable_count) < 3000 else 'IMBALANCED - SHUFFLING NEEDED!'}")
