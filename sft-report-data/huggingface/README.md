---
base_model: unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- unsloth
- qwen2.5-coder
- manim
---

# Chimanwakis/qwen_manim_animation_q4_k_m_v5

GGUF export of a Manim-specialized SFT model. The LoRA adapter was merged into
`unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` before conversion with Unsloth.

Training dataset: `primary_maths_manim_qwen_messages_final_6023.jsonl`.

Dataset SHA-256: `1a784d85752316042438700de8c83ab5d27d246cb990b09789ac88dd34cc8fbf`.

## Files

| File | Size |
|---|---:|
| `qwen2.5-coder-3b-instruct.Q4_K_M.gguf` | 1.80 GiB |

Quantization requested: `Q4_K_M`. File hashes are available in
`SHA256SUMS`.

## llama.cpp

```bash
llama-cli -m qwen2.5-coder-3b-instruct.Q4_K_M.gguf -cnv -c 4096
```

Increase `-c` only when the computer has enough memory.
