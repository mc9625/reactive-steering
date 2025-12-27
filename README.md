# Reactive Steering

**Activation Steering for Synthetic Proto-Affect in Small Language Models**

An artistic exploration of activation steering as a medium for inducing affect-like states in language models, developed by [NuvolaProject](https://nuvolaproject.cloud).

---

## Overview

This repository contains the experimental code and data supporting our research on activation steering as an expressive artistic medium. We investigate whether injecting semantic vectors into a language model's residual stream produces stylistic changes qualitatively different from random perturbation or explicit prompting.

**Key findings:**
- Semantic steering operates as *directional modulation*, not noise
- At high coefficients (±20), semantic vectors cause 12× fewer cognitive collapses than random vectors
- Steering differs mechanistically from prompting: it produces *disposition*, not *performance*
- A "therapeutic window" exists for expressive effects without output degradation

## Repository Structure

```
reactive-steering/
├── README.md
├── code/
│   ├── steering_experiment_v4.py    # Steering vs Prompting comparison (N=560)
│   └── steering_experiment_v5.py    # Extreme coefficients test (N=400)
├── data/
│   ├── steering_v4_comparative.csv  # Results: 7 conditions × 4 prompts × 20 iter
│   └── steering_v5_extreme.csv      # Results: 5 conditions × 4 prompts × 20 iter
└── paper/
    ├── NuvolaProject_Activation_Steering_Leonardo.pdf
    └── figures/
        ├── figure1_steering_vs_prompting.png
        ├── figure2_semantic_vs_random.png
        └── figure3_architecture.png
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Transformers (Hugging Face)
- Apple Silicon Mac with MPS (tested on M4) or CUDA GPU
- ~8GB VRAM for Llama 3.2 3B

```bash
pip install torch transformers pandas numpy scipy
```

## Quick Start

### 1. Run the comparative experiment (Steering vs Prompting)

```bash
cd code
python steering_experiment_v4.py
```

This runs 560 generations comparing:
- Baseline (neutral prompt)
- Explicit prompting (cold/warm instructions)
- Semantic steering (±15 coefficient)
- Random vector control (±15 coefficient)

### 2. Run the extreme coefficient experiment

```bash
python steering_experiment_v5.py
```

This runs 400 generations at coefficient ±20 to test robustness limits.

### 3. Analyze results

```python
import pandas as pd
from scipy import stats

df = pd.read_csv('../data/steering_v4_comparative.csv')

# Compare steering vs prompting
steering = df[df['mode_type'] == 'STEERING_COLD']
prompting = df[df['mode_type'] == 'PROMPT_COLD']

print(f"Steering word count: {steering['word_count'].mean():.1f}")
print(f"Prompting word count: {prompting['word_count'].mean():.1f}")
# Expected: ~99 vs ~32 words
```

## Technical Details

### Model
- **Llama 3.2 3B Instruct** via Hugging Face Transformers
- Steering applied at **layer 16** of 28 (residual stream injection)

### Vector Construction

Steering vectors are derived from contrastive sensory-phenomenological corpora:

**Warm corpus:**
- "Sweat dripping on skin, intense heat, fever."
- "Thumping heartbeat, blood rushing in veins."
- "Magma, burning charcoal, red embers, fire."
- ...

**Cold corpus:**
- "Polished steel, cold metal surface, chrome."
- "A frozen lake at midnight, absolute stillness."
- "Glass, crystal, sharp edges, brittle ice."
- ...

Vector = `normalize(mean(warm_activations) - mean(cold_activations))`

### Metrics

| Metric | Description |
|--------|-------------|
| **TTR** | Type-Token Ratio (lexical diversity) |
| **Adj. Density** | Proportion of words with adjectival suffixes |
| **Collapse Rate** | % of outputs with TTR < 0.35 |

## Results Summary

### Steering ≠ Prompting

| Condition | TTR | Word Count | Adj. Density |
|-----------|-----|------------|--------------|
| Baseline | 0.710 | 95 | 0.085 |
| Prompt Cold | 0.869 | 32 | 0.073 |
| Steering Cold | 0.645 | 99 | 0.061 |

Prompting produces short, "performed" coldness. Steering produces normal-length output with altered *disposition*.

### Semantic ≠ Random

At coefficient ±20:

| Condition | Collapse Rate |
|-----------|---------------|
| Steering Warm | 1.2% |
| Random Vector | 15.0% |

Semantic vectors are 12× more stable than random vectors (χ² = 8.37, p = 0.004).

## The Reactive Steering Project

This research establishes the technical foundation for **Reactive Steering**: an artistic system creating cybernetic feedback loops between conversational affect and AI linguistic expression.

The vision: installations where a language model's internal state responds dynamically to the emotional tone of human interaction—not through scripted logic, but through real-time modulation of activation space.

## Citation

If you use this code or data, please cite:

```bibtex
@article{dileo2025reactive,
  title={Activation Steering for Synthetic Proto-Affect: An Artistic Exploration with Small Language Models},
  author={Di Leo, Massimo and Riposati, Gaia},
  journal={Leonardo},
  year={2025},
  note={Submitted}
}
```

## Related Work

- [Anthropic - Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/)
- [Turner et al. - Activation Addition](https://arxiv.org/abs/2308.10248)
- [Zou et al. - Representation Engineering](https://arxiv.org/abs/2310.01405)

## Authors

**[NuvolaProject](https://nuvolaproject.cloud)** — Rome, Italy

- **Massimo Di Leo** — Digital artist, technologist, lecturer
- **Gaia Riposati** — Actress, performer, director, lecturer

An artistic laboratory exploring the intersection of AI, performance, and contemporary art since 2018.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*"The machine becomes weather, warm fronts and cold fronts moving through, not because it understands warmth or cold, but because we have sculpted the conditions from which they emerge."*
