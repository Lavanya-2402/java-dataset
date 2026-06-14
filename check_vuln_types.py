import json
from collections import Counter

vuln_types = Counter()

with open('train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        vuln_type = data.get('vulnerability_type', 'Unknown')
        input_code = data.get('input', '').strip()
        output = (data.get('completion', '') or data.get('output', '') or '').strip()
        
        # Only count vulnerable examples
        if input_code != output:
            vuln_types[vuln_type] += 1

print(f"Total Unique Vulnerability Types: {len(vuln_types)}\n")
print("Breakdown:")
for vuln, count in sorted(vuln_types.items(), key=lambda x: -x[1]):
    print(f"  {vuln}: {count}")
