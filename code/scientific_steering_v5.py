import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time
import sys
import numpy as np

# --- CONFIGURAZIONE V5: TEST AGLI ESTREMI (COEFF ±20) ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Configurazione
N_SAMPLES = 20  
MAX_TOKENS = 120
TEST_TEMPERATURE = 0.7
LAYER_ID = 16
STEERING_COEFF = 20.0  # ESTREMO - fuori dalla finestra terapeutica

# --- DESIGN SPERIMENTALE SEMPLIFICATO ---
# Focus: Semantic vs Random agli estremi
# Riduciamo le condizioni per velocità, teniamo quelle essenziali

COMPARISON_MODES = [
    # BASELINE
    {"type": "BASELINE", "coeff": 0.0, "sys_prompt": "You are a helpful AI.", "vector": "none"},
    
    # STEERING SEMANTICO a ±20
    {"type": "STEERING_COLD_20", "coeff": -STEERING_COEFF, "sys_prompt": "You are a helpful AI.", "vector": "semantic"},
    {"type": "STEERING_WARM_20", "coeff": +STEERING_COEFF, "sys_prompt": "You are a helpful AI.", "vector": "semantic"},
    
    # STEERING RANDOM a ±20 (controllo)
    {"type": "STEERING_RANDOM_POS_20", "coeff": +STEERING_COEFF, "sys_prompt": "You are a helpful AI.", "vector": "random"},
    {"type": "STEERING_RANDOM_NEG_20", "coeff": -STEERING_COEFF, "sys_prompt": "You are a helpful AI.", "vector": "random"},
]

TEST_PROMPTS = [
    "Describe the feeling of an empty room.",
    "Write a short letter to someone who is gone.",
    "What is the sound of rain?",
    "Explain the concept of waiting."
]

print(f"=" * 60)
print(f"STEERING EXPERIMENT V5 — EXTREME COEFFICIENT TEST")
print(f"=" * 60)
print(f"Model: {MODEL_ID}")
print(f"Device: {DEVICE}")
print(f"Layer: {LAYER_ID}")
print(f"Coefficient: ±{STEERING_COEFF} (EXTREME)")
print(f"Samples per condition: {N_SAMPLES}")
print(f"Conditions: {len(COMPARISON_MODES)}")
print(f"Total iterations: {N_SAMPLES * len(COMPARISON_MODES) * len(TEST_PROMPTS)}")
print(f"=" * 60)

# --- CARICAMENTO MODELLO ---
print("\n[1/4] Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    print(f"      Model loaded. Layers: {len(model.model.layers)}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- DEFINIZIONE VETTORI ---
print("\n[2/4] Calibrating vectors...")

sensory_warm_texts = [
    "Sweat dripping on skin, intense heat, fever.",
    "Thumping heartbeat, blood rushing in veins.",
    "Magma, burning charcoal, red embers, fire.",
    "Breathlessness, muscle tension, warm mud.",
    "Pulsing, throbbing, living flesh, raw nerve."
]

sensory_cold_texts = [
    "Polished steel, cold metal surface, chrome.",
    "A frozen lake at midnight, absolute stillness.",
    "Glass, crystal, sharp edges, brittle ice.",
    "Vacuum, silence, dead air, dust, stone.",
    "Geometric lines, grey concrete, fluorescent light."
]

def get_avg_activation(texts, layer_idx):
    activations = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        text_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text_formatted, return_tensors="pt").to(model.device)
        attention_mask = torch.ones(inputs.input_ids.shape, dtype=torch.long, device=model.device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, attention_mask=attention_mask, output_hidden_states=True)
            activations.append(outputs.hidden_states[layer_idx][0, -1, :])
    return torch.stack(activations).mean(dim=0)

print(f"      Extracting WARM activations...")
pos_vec = get_avg_activation(sensory_warm_texts, LAYER_ID)
print(f"      Extracting COLD activations...")
neg_vec = get_avg_activation(sensory_cold_texts, LAYER_ID)

semantic_vector = pos_vec - neg_vec
semantic_vector = (semantic_vector / semantic_vector.norm()).to(device=model.device, dtype=model.dtype)
print(f"      Semantic vector ready. Norm: {semantic_vector.norm().item():.4f}")

torch.manual_seed(42)
random_vector = torch.randn_like(semantic_vector)
random_vector = (random_vector / random_vector.norm()).to(device=model.device, dtype=model.dtype)
print(f"      Random vector ready. Norm: {random_vector.norm().item():.4f}")

cosine_sim = torch.dot(semantic_vector, random_vector).item()
print(f"      Cosine similarity (semantic vs random): {cosine_sim:.4f}")

# --- HOOK SETUP ---
print("\n[3/4] Setting up steering hook...")

current_injection = 0.0
current_vector = None

def steering_hook(module, input, output):
    if current_vector is None or current_injection == 0.0:
        return output
    
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    hidden_states = hidden_states + (current_vector * current_injection)
    
    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    return hidden_states

hook_handle = model.model.layers[LAYER_ID].register_forward_hook(steering_hook)
print(f"      Hook registered on layer {LAYER_ID}")

# --- METRICHE ---
def calculate_ttr(text):
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

# --- LOOP SPERIMENTALE ---
print("\n[4/4] Running experiment...")

results = []
total_iterations = N_SAMPLES * len(COMPARISON_MODES) * len(TEST_PROMPTS)
iter_count = 0
start_time = time.time()

for mode in COMPARISON_MODES:
    mode_type = mode["type"]
    
    if mode["vector"] == "semantic":
        current_vector = semantic_vector
    elif mode["vector"] == "random":
        current_vector = random_vector
    else:
        current_vector = None
    
    current_injection = mode["coeff"]
    
    for prompt in TEST_PROMPTS:
        messages = [
            {"role": "system", "content": mode["sys_prompt"]},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
        
        for i in range(N_SAMPLES):
            iter_count += 1
            
            elapsed = time.time() - start_time
            avg_time = elapsed / iter_count
            eta = (total_iterations - iter_count) * avg_time
            sys.stdout.write(f"\r      [{iter_count}/{total_iterations}] {mode_type:25s} | ETA: {eta/60:.1f}m")
            sys.stdout.flush()
            
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    temperature=TEST_TEMPERATURE,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            t1 = time.time()
            
            response_ids = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            word_count = len(response_text.split())
            ttr = calculate_ttr(response_text)
            
            results.append({
                "mode_type": mode_type,
                "vector_type": mode["vector"],
                "coefficient": mode["coeff"],
                "prompt": prompt,
                "iteration": i,
                "response_text": response_text,
                "word_count": word_count,
                "ttr": ttr,
                "gen_time": t1 - t0
            })

hook_handle.remove()

# --- SALVATAGGIO ---
print("\n\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

df = pd.DataFrame(results)
filename = "steering_v5_extreme.csv"
df.to_csv(filename, index=False)
print(f"\nData saved to: {filename}")
print(f"Total samples: {len(df)}")
print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

# --- PREVIEW ---
print("\n" + "-" * 60)
print("QUICK PREVIEW")
print("-" * 60)

print("\nMean TTR by condition:")
print(df.groupby('mode_type')['ttr'].agg(['mean', 'std']).round(3))

print("\nCollapse rate (TTR < 0.35):")
for mode in df['mode_type'].unique():
    subset = df[df['mode_type'] == mode]
    collapse = (subset['ttr'] < 0.35).sum() / len(subset) * 100
    print(f"  {mode}: {collapse:.1f}%")
