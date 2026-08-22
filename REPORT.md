# Technical Report — Offline Calculus Animation Assistant

**Team ID:** TODO_REPLACE_TEAM_ID  
**Domain:** math_scientific_reasoning  
**Model:** Qwen-Calculus-SFT-Q4_K_M

---

## Problem

This submission targets learners and educators who need clear, visual calculus explanations but may not have reliable internet access or high-end computing hardware. The model is designed to generate mathematical explanations and executable Manim Community animation code locally on a budget laptop.

TODO: Add the specific target users, African context, deployment setting, and evidence that motivates the problem.

---

## Design Decisions

- **Fine-tuned model:** `Chimanwakis/qwen_manim_animation_16bit`
- **Submission weights:** `qwen_manim_animation_16bit.Q4_K_M.gguf`
- **Architecture:** Qwen2-family model, approximately 3.09B parameters
- **Quantization:** GGUF Q4_K_M
- **Runtime:** llama.cpp
- **Model file size:** approximately 1.93 GB
- **Training dataset:** `Chimanwakis/calculus_manim`
- **Hosting:** public Hugging Face repository `Chimanwakis/qwen-calculus-sft-GGUF`

Q4_K_M was selected to balance mathematical and code-generation quality with the strict 8 GB RAM evaluation limit.

TODO: Describe the fine-tuning method, dataset construction and validation, rejected alternatives, and any observed trade-offs.

---

## Constraints

- Target profile: 4 vCPU, 8 GB RAM, integrated GPU only
- Inference must run entirely offline through llama.cpp
- The model and context cache must remain below the evaluation machine's memory limit
- The use case must remain useful where internet access is unavailable, slow, or expensive

TODO: Add the development hardware, context length used for testing, inference flags, and connectivity or power constraints.

---

## Benchmarks

| Metric | Value |
|---|---|
| Development machine | TODO |
| llama.cpp version/commit | TODO |
| Context size | TODO |
| Peak RAM | TODO |
| Time to first token | TODO |
| Prompt processing speed | TODO |
| Generation speed | TODO |
| Thermal throttling | TODO |

TODO: Run the ADTC profiler and replace every benchmark placeholder with measured results. These values are self-reported; official scores will be measured on the standard evaluation machine.
