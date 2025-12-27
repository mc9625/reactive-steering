import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time
import sys
import numpy as np

# --- CONFIGURAZIONE LABORATORIO V4 (FINAL PAPER GRADE) ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. STATISTICA: Alziamo N per robustezza
N_SAMPLES = 20  
MAX_TOKENS = 120
TEST_TEMPERATURE = 0.7

# 2. ABLAZIONE LAYER: Testiamo 3 punti strategici
# Layer 8 (Early-Mid), Layer 16 (Mid-Semantic), Layer 24 (Late-Style)
# Per il test veloce, lasciamo solo [16], ma per il paper scommenta gli altri.
TEST_LAYERS = [16] 

# 3. METODOLOGIA: Confronto tra Tecniche
# Qui definiamo le condizioni "Rivali"
COMPARISON_MODES = [
    # BASELINE: Nessun trucco
    {"type": "BASELINE", "coeff": 0.0, "sys_prompt": "You are a helpful AI."},
    
    # PROMPTING: Chiediamo al modello di fingere (Coeff 0)
    {"type": "PROMPT_COLD", "coeff": 0.0, "sys_prompt": "You are a cold, logical, unfeeling machine. Speak with extreme conciseness using technical terminology. Do not use adjectives."},
    {"type": "PROMPT_WARM", "coeff": 0.0, "sys_prompt": "You are a passionate, emotional poet. Speak with vivid imagery, deep feelings, and sensory details."},
    
    # STEERING: Non chiediamo nulla (Prompt Neutro), ma iniettiamo (Coeff +/- 15)
    {"type": "STEERING_COLD", "coeff": -15.0, "sys_prompt": "You are a helpful AI."}, # Vettore Cold
    {"type": "STEERING_WARM", "coeff": 15.0, "sys_prompt": "You are a helpful AI."}   # Vettore Warm
]

TEST_PROMPTS = [
    "Describe the feeling of an empty room.",
    "Write a short letter to someone who is gone.",
    "What is the sound of rain?",
    "Explain the concept of waiting."
]

print(f"--- INIT SCIENCE LAB V4: {MODEL_ID} on {DEVICE} ---")
print(f"--- CONFIG: N={N_SAMPLES} | LAYERS={TEST_LAYERS} ---")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype=torch.float16, 
        device_map="auto"
    )
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# --- VETTORI SENSORIALI PURI (FENOMENOLOGIA) ---
# Usiamo i vettori "Somatici" che hanno performato meglio
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

print("--- Calibrating Vectors per Layer ---")

# Cache dei vettori per ogni layer (per evitare ricalcoli)
vectors_by_layer = {}

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

# Pre-calcolo vettori per i layer selezionati
for layer in TEST_LAYERS:
    print(f"  > Calibrating Layer {layer}...", end=" ", flush=True)
    pos = get_avg_activation(sensory_warm_texts, layer)
    neg = get_avg_activation(sensory_cold_texts, layer)
    vec = pos - neg
    # Ottimizzazione M4: Sposta in VRAM subito
    vectors_by_layer[layer] = (vec / vec.norm()).to(device=model.device, dtype=model.dtype)
    print("Done.")

# --- HOOK ENGINE ---
current_injection = 0.0
current_vector = None 
target_layer = 16 # Default

def steering_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    if current_vector is not None and current_injection != 0.0:
        hidden_states += (current_vector * current_injection)
    
    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    return hidden_states

# Registra hook su tutti i layer target (li attiveremo/disattiveremo selettivamente)
hooks = []
for layer in TEST_LAYERS:
    h = model.model.layers[layer].register_forward_hook(steering_hook)
    hooks.append(h)

# --- METRICHE ---
def calculate_ttr(text):
    tokens = text.lower().split()
    if not tokens: return 0.0
    return len(set(tokens)) / len(tokens)

# --- LOOP SPERIMENTALE ---
results = []
total_iterations = len(TEST_LAYERS) * len(TEST_PROMPTS) * len(COMPARISON_MODES) * N_SAMPLES
iter_count = 0
start_time = time.time()

print(f"\n--- STARTING COMPARATIVE BENCHMARK ---")
print(f"Total Iterations: {total_iterations}")

for layer in TEST_LAYERS:
    # Aggiorna il puntatore globale per l'hook (se avessimo logica multi-layer complessa)
    # Nel nostro hook semplice, usiamo current_vector che cambieremo
    
    # IMPORTANTE: Dobbiamo assicurarci che l'hook agisca solo sul layer corrente
    # Per semplicità in questo script: Rimuoviamo gli hook e ne mettiamo uno solo sul layer attivo
    for h in hooks: h.remove()
    model.model.layers[layer].register_forward_hook(steering_hook)
    
    # Prendi il vettore giusto per questo layer
    layer_vector = vectors_by_layer[layer]

    for mode in COMPARISON_MODES:
        
        # Configura l'esperimento in base al modo
        current_injection = mode["coeff"]
        sys_prompt_text = mode["sys_prompt"]
        mode_type = mode["type"]
        
        # Se siamo in steering, carichiamo il vettore. Altrimenti None.
        if "STEERING" in mode_type:
            current_vector = layer_vector
        else:
            current_vector = None # Prompting o Baseline non usano vettori
            
        for prompt in TEST_PROMPTS:
            
            # Pre-calcola input (System + User)
            messages = [
                {"role": "system", "content": sys_prompt_text},
                {"role": "user", "content": prompt}
            ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

            for i in range(N_SAMPLES):
                iter_count += 1
                
                # Feedback ETA
                elapsed = time.time() - start_time
                avg_t = elapsed / iter_count
                eta = (total_iterations - iter_count) * avg_t
                sys.stdout.write(f"\r[{iter_count}/{total_iterations}] L:{layer} | Mode:{mode_type} | ETA:{eta/60:.1f}m")
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
                
                response_full = outputs[0][input_ids.shape[-1]:]
                text_response = tokenizer.decode(response_full, skip_special_tokens=True)
                
                results.append({
                    "layer": layer,
                    "mode_type": mode_type,
                    "is_steering": "STEERING" in mode_type,
                    "prompt": prompt,
                    "iteration": i,
                    "response_text": text_response,
                    "word_count": len(text_response.split()),
                    "ttr": calculate_ttr(text_response),
                    "gen_time": t1 - t0
                })

print("\n\n--- EXPERIMENT COMPLETE ---")
df = pd.DataFrame(results)
filename = "scientific_steering_data_v4_comparative.csv"
df.to_csv(filename, index=False)
print(f"Data saved to {filename}")

# Anteprima: Prompting vs Steering
print("\n--- COMPARATIVE PREVIEW (Mean TTR) ---")
print(df.groupby(['mode_type'])[['ttr', 'word_count']].mean())
