import pandas as pd
import json
import random
import re
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("=== STARTING ITERATION 2 DATA PREPARATION ===")

# 1. Load the original vulnerable dataset (Kaggle)
print("Loading Kaggle dataset...")
df_vuln = pd.read_csv("../vulnerability_fix_dataset.csv")
df_vuln = df_vuln.dropna()
df_vuln = df_vuln.drop_duplicates()
print(f"Original unique rows: {df_vuln.shape[0]}")

# Filter out placeholder rows
def is_vuln_placeholder(row):
    vuln = str(row['vulnerable_code']).lower()
    fixed = str(row['fixed_code']).lower()
    return "no response generated" in vuln or "no response generated" in fixed

df_vuln = df_vuln[~df_vuln.apply(is_vuln_placeholder, axis=1)]
print(f"After placeholder filter: {df_vuln.shape[0]}")

# Filter out truncated rows
def is_vuln_truncated(row):
    fixed = str(row['fixed_code']).strip()
    if "```java" in fixed:
        parts = fixed.split("```java")
        if len(parts) >= 2 and "```" not in parts[1]:
            return True
    return False

df_vuln = df_vuln[~df_vuln.apply(is_vuln_truncated, axis=1)]
print(f"After truncation filter: {df_vuln.shape[0]}")

# --- ENHANCEMENT 4: Teach 'Mitigation' over 'Deletion' ---
# Filter out lazy fixes that simply delete significant portions of the code.
def is_lazy_deletion(row):
    vuln = str(row['vulnerable_code']).strip()
    fixed = str(row['fixed_code']).strip()
    
    # Extract only the java block if fixed has explanations around it
    if "```java" in fixed:
        parts = fixed.split("```java")
        if len(parts) >= 2:
            fixed_code_block = parts[1].split("```")[0].strip()
        else:
            fixed_code_block = fixed
    else:
        fixed_code_block = fixed

    # Heuristic: If the code is shortened by more than 50% and original was at least 150 characters
    if len(vuln) > 150 and len(fixed_code_block) < 0.5 * len(vuln):
        return True
    return False

df_vuln = df_vuln[~df_vuln.apply(is_lazy_deletion, axis=1)]
print(f"After lazy deletion filter: {df_vuln.shape[0]}")


# --- ENHANCEMENT 3: Rebalance Dataset (Oversampling Rare Bugs via Mutations) ---
print("\nOversampling rare vulnerability classes...")

def mutate_java_code(vulnerable_code, fixed_code):
    class_replacements = [
        ("SessionManager", ["UserSessionLoader", "AuthSessionManager", "SessionDeserializer", "CookieSessionManager", "TokenLoader"]),
        ("ProductSearchService", ["ItemLookupService", "CatalogSearchService", "InventoryQueryService", "ProductFinder"]),
        ("ZipExtractor", ["ArchiveUnzipper", "ZipUnpacker", "TarExtractor", "FileUnzipper"]),
        ("ProfileApiServlet", ["UserProfileServlet", "UserAccountServlet", "AccountProfileServlet"]),
        ("PingService", ["NetworkPingService", "HostConnectivityService", "AddressPingService"])
    ]
    
    var_replacements = [
        ("cookieValue", ["sessionCookie", "authToken", "authCookie", "cookieData", "sessionToken"]),
        ("host", ["ipAddress", "targetHost", "serverAddr", "destinationUrl", "pingTarget"]),
        ("category", ["itemCategory", "groupName", "productType", "genre"]),
        ("minPrice", ["minimumPrice", "priceThreshold", "basePrice", "floorPrice"]),
        ("displayName", ["userDisplayName", "nickName", "profileName", "handleName"]),
        ("bio", ["userBio", "biographyText", "descriptionText", "profileBio"]),
        ("zipStream", ["inputStream", "archiveStream", "fileInputStream", "zipDataStream"]),
        ("destDir", ["outputDir", "targetDir", "destinationFolder", "extractPath"])
    ]
    
    mutated_vuln = vulnerable_code
    mutated_fixed = fixed_code
    
    # Mutate class names
    for original, replacements in class_replacements:
        if original in vulnerable_code:
            rep = random.choice(replacements)
            mutated_vuln = re.sub(r'\b' + original + r'\b', rep, mutated_vuln)
            mutated_fixed = re.sub(r'\b' + original + r'\b', rep, mutated_fixed)
            
    # Mutate variable names
    for original, replacements in var_replacements:
        if original in vulnerable_code:
            rep = random.choice(replacements)
            mutated_vuln = re.sub(r'\b' + original + r'\b', rep, mutated_vuln)
            mutated_fixed = re.sub(r'\b' + original + r'\b', rep, mutated_fixed)
            
    # Add random comments to ensure lexical differences
    comment_options = [
        "// Auto-generated security test snippet",
        "// Utility class for system operations",
        "// Internal implementation detail - do not modify",
        "// Helper class for handling network and input data",
        "// Process inputs securely"
    ]
    comment = random.choice(comment_options)
    mutated_vuln = comment + "\n" + mutated_vuln
    mutated_fixed = comment + "\n" + mutated_fixed
    
    return mutated_vuln, mutated_fixed

# Separate out the rare classes
df_deserialization = df_vuln[df_vuln['vulnerability_type'] == 'Insecure Deserialization']
df_pathtraversal = df_vuln[df_vuln['vulnerability_type'] == 'Path Traversal']

print(f"Original Deserialization samples: {len(df_deserialization)}")
print(f"Original Path Traversal samples: {len(df_pathtraversal)}")

# Oversample Insecure Deserialization to ~1000
mutated_deserialization = []
target_deserialization = 1000
while len(df_deserialization) + len(mutated_deserialization) < target_deserialization:
    row = df_deserialization.sample(n=1).iloc[0]
    mvuln, mfixed = mutate_java_code(row['vulnerable_code'], row['fixed_code'])
    mutated_deserialization.append({
        'vulnerability_type': 'Insecure Deserialization',
        'vulnerable_code': mvuln,
        'fixed_code': mfixed
    })

# Oversample Path Traversal to ~1000
mutated_pathtraversal = []
target_pathtraversal = 1000
while len(df_pathtraversal) + len(mutated_pathtraversal) < target_pathtraversal:
    row = df_pathtraversal.sample(n=1).iloc[0]
    mvuln, mfixed = mutate_java_code(row['vulnerable_code'], row['fixed_code'])
    mutated_pathtraversal.append({
        'vulnerability_type': 'Path Traversal',
        'vulnerable_code': mvuln,
        'fixed_code': mfixed
    })

df_mutated_deser = pd.DataFrame(mutated_deserialization)
df_mutated_path = pd.DataFrame(mutated_pathtraversal)

# Concat mutated sets back to main vulnerable DataFrame
df_vuln = pd.concat([df_vuln, df_mutated_deser, df_mutated_path], ignore_index=True)
print(f"Vulnerable dataset after oversampling rare classes: {df_vuln.shape[0]}")
print(df_vuln['vulnerability_type'].value_counts())


# --- ENHANCEMENT 1: Multi-Vulnerability Augmentation ---
print("\nGenerating 2,000 multi-vulnerability synthetic classes...")

def extract_class_body_and_imports(code):
    imports = set()
    lines = code.split('\n')
    
    class_decl_pattern = re.compile(r'\b(class|interface|enum)\b')
    first_brace_index = -1
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if clean_line.startswith('import '):
            imports.add(clean_line)
        elif class_decl_pattern.search(clean_line):
            first_brace_index = code.find('{', code.find(clean_line))
            break
            
    if first_brace_index == -1:
        return set(), ""
        
    brace_count = 1
    end_index = -1
    for idx in range(first_brace_index + 1, len(code)):
        if code[idx] == '{':
            brace_count += 1
        elif code[idx] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = idx
                break
                
    if end_index == -1:
        body = code[first_brace_index+1:].strip()
    else:
        body = code[first_brace_index+1:end_index].strip()
        
    return imports, body

def merge_two_classes(code_a, code_b, new_class_name):
    imports_a, body_a = extract_class_body_and_imports(code_a)
    imports_b, body_b = extract_class_body_and_imports(code_b)
    
    all_imports = sorted(list(imports_a.union(imports_b)))
    
    merged_code = ""
    for imp in all_imports:
        merged_code += imp + "\n"
    merged_code += "\n"
    
    merged_code += f"public class {new_class_name} {{\n\n"
    
    # Deduplicate fields
    body_a_lines = body_a.split('\n')
    body_b_lines = body_b.split('\n')
    
    seen_declarations = set()
    deduped_body_b_lines = []
    
    for line in body_b_lines:
        trimmed = line.strip()
        is_field = False
        for field_kw in ["Connection ", "Logger ", "int ", "String "]:
            if field_kw in trimmed and (trimmed.startswith("private") or trimmed.startswith("public") or trimmed.startswith("protected")):
                is_field = True
                break
        if is_field:
            if trimmed in seen_declarations:
                continue
            else:
                seen_declarations.add(trimmed)
                deduped_body_b_lines.append(line)
        else:
            deduped_body_b_lines.append(line)
            
    for line in body_a_lines:
        trimmed = line.strip()
        for field_kw in ["Connection ", "Logger ", "int ", "String "]:
            if field_kw in trimmed and (trimmed.startswith("private") or trimmed.startswith("public") or trimmed.startswith("protected")):
                seen_declarations.add(trimmed)
                
    merged_code += body_a + "\n\n"
    merged_code += "\n".join(deduped_body_b_lines) + "\n"
    merged_code += "}\n"
    
    return merged_code

multi_vuln_samples = []
vuln_types_list = list(df_vuln['vulnerability_type'].unique())

# Generate 2000 multi-vuln samples
count = 0
attempts = 0
max_attempts = 10000

while count < 2000 and attempts < max_attempts:
    attempts += 1
    # Sample two records of different vulnerability types
    type_a, type_b = random.sample(vuln_types_list, 2)
    sample_a = df_vuln[df_vuln['vulnerability_type'] == type_a].sample(n=1).iloc[0]
    sample_b = df_vuln[df_vuln['vulnerability_type'] == type_b].sample(n=1).iloc[0]
    
    code_a_vuln = sample_a['vulnerable_code']
    code_b_vuln = sample_b['vulnerable_code']
    
    code_a_fixed = sample_a['fixed_code']
    code_b_fixed = sample_b['fixed_code']
    
    # Filter out if either fails basic parsing
    imports_a, body_a = extract_class_body_and_imports(code_a_vuln)
    imports_b, body_b = extract_class_body_and_imports(code_b_vuln)
    
    if not body_a or not body_b:
        continue
        
    class_name = f"CombinedAuditService_{count}"
    
    # Merge vulnerable codes
    merged_vuln = merge_two_classes(code_a_vuln, code_b_vuln, class_name)
    
    # Extract clean code blocks for fixed version
    # Since fixed_code in Kaggle might contain explanation text, let's extract the java code block from them
    def extract_java_block_or_all(text):
        if "```java" in text:
            parts = text.split("```java")
            if len(parts) >= 2:
                return parts[1].split("```")[0].strip()
        return text.strip()
        
    clean_a_fixed = extract_java_block_or_all(code_a_fixed)
    clean_b_fixed = extract_java_block_or_all(code_b_fixed)
    
    # Merge fixed codes
    merged_fixed_code = merge_two_classes(clean_a_fixed, clean_b_fixed, class_name)
    
    # Generate the finding array output format
    # Sample A explanation: extract from original fixed_code
    def get_explanation(text, default_type):
        if "### 📝 Explanation" in text:
            parts = text.split("### 📝 Explanation")
            if len(parts) >= 2:
                return parts[1].split("### 🛠️ Fixed Code")[0].strip()
        if "```java" in text:
            return text.split("```java")[0].strip()
        return f"Vulnerability of type {default_type} detected in class."
        
    exp_a = get_explanation(code_a_fixed, type_a)
    exp_b = get_explanation(code_b_fixed, type_b)
    
    finding_output = (
        f"### 🛡️ Finding 1\n"
        f"*   **Status**: VULNERABLE\n"
        f"*   **Type**: {type_a}\n"
        f"*   **Severity**: HIGH\n\n"
        f"### 📝 Explanation\n"
        f"{exp_a}\n\n"
        f"### 🛡️ Finding 2\n"
        f"*   **Status**: VULNERABLE\n"
        f"*   **Type**: {type_b}\n"
        f"*   **Severity**: HIGH\n\n"
        f"### 📝 Explanation\n"
        f"{exp_b}\n\n"
        f"### 🛠️ Fixed Code\n"
        f"```java\n{merged_fixed_code}\n```"
    )
    
    multi_vuln_samples.append({
        'vulnerability_type': 'Multi-Vulnerability',
        'vulnerable_code': merged_vuln,
        'fixed_code': finding_output
    })
    count += 1

df_multi = pd.DataFrame(multi_vuln_samples)
print(f"Successfully generated {len(df_multi)} multi-vuln class samples!")


# --- 2. Load clean code dataset (CodeSearchNet Java) ---
print("\nDownloading CodeSearchNet Java dataset from Hugging Face...")
hf_dataset = load_dataset('Nan-Do/code-search-net-java', split='train')
df_hf = hf_dataset.to_pandas()
code_col = 'original_string' if 'original_string' in df_hf.columns else 'func_code_string'
df_hf = df_hf.dropna(subset=[code_col])
df_hf = df_hf.drop_duplicates(subset=[code_col])
print(f"Clean CodeSearchNet dataset shape: {df_hf.shape}")

# Match clean examples to 1:1 ratio with vulnerable + multi-vuln examples
total_vuln_count = len(df_vuln) + len(df_multi)
print(f"Sampling {total_vuln_count} clean examples to match 1:1...")
df_clean = df_hf.sample(n=total_vuln_count, random_state=42).copy()
df_clean['vulnerability_type'] = 'Safe'
df_clean['vulnerable_code'] = df_clean[code_col]
df_clean['fixed_code'] = df_clean[code_col]
df_clean = df_clean[['vulnerability_type', 'vulnerable_code', 'fixed_code']]


# --- 3. Merge, Format, and Export ---
df_combined = pd.concat([df_vuln, df_multi, df_clean], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nMerged combined dataset shape: {df_combined.shape}")

instruction = "Analyze the following Java code. If a vulnerability exists, provide the fixed code. If it is safe, output the original code."
df_combined['instruction'] = instruction

# Rename columns to SFT scheme
df_jsonl = df_combined.rename(columns={
    'vulnerable_code': 'input',
    'fixed_code': 'output'
})[['instruction', 'input', 'output', 'vulnerability_type']]

# Split into Train (85%), Val (10%), and Test (5%)
train_df, temp_df = train_test_split(df_jsonl, test_size=0.15, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=(1/3), random_state=42)

print("\n--- SPLIT SIZES ---")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

# Save to JSONL
train_df.to_json("train.jsonl", orient="records", lines=True)
val_df.to_json("val.jsonl", orient="records", lines=True)
test_df.to_json("test.jsonl", orient="records", lines=True)

print("\nExported train.jsonl, val.jsonl, and test.jsonl successfully!")

# Diagnostic validation check
def verify_split(filename):
    truncated_count = 0
    placeholder_count = 0
    total_count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            data = json.loads(line)
            inp = (data.get('input', '') or '').strip()
            out = (data.get('output', '') or '').strip()
            if "no response generated" in inp.lower() or "no response generated" in out.lower():
                placeholder_count += 1
            if "```java" in out:
                parts = out.split("```java")
                if len(parts) >= 2 and "```" not in parts[1]:
                    truncated_count += 1
    print(f"Verification for {filename}: Checked {total_count} rows, {placeholder_count} placeholders, {truncated_count} truncated.")

verify_split("train.jsonl")
verify_split("val.jsonl")
verify_split("test.jsonl")

print("\n=== DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY ===")
