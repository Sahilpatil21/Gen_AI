# LoRA / QLoRA Experiment Plan

## Research question
Does parameter-efficient fine-tuning improve the response quality of the personalized assistant compared with the base model, and how does it compare with RAG alone?

## Suggested conditions
1. Base model without RAG.
2. Base model with RAG.
3. LoRA-adapted model without RAG.
4. LoRA-adapted model with RAG.
5. Optional: QLoRA-adapted model with RAG.

Keep model version, test questions, decoding settings, and scoring rubric fixed.

## Dataset
`data/training/train.jsonl` is a tiny format example only. Replace or expand it with examples you have permission to use. Each line has `instruction`, optional `input`, and `output`.

## Evaluation rubric
Score each answer from 1 to 5 on correctness, relevance, completeness, grounding in retrieved documents, and source usefulness. Also record unsupported claims (lower is better). Include questions answered by the documents and questions whose answers are absent. Use a held-out test set not used in training. A second reviewer scoring some answers can reduce subjectivity.

| Condition | Correctness (1–5) | Relevance (1–5) | Grounding (1–5) | Unsupported claims | Notes |
|---|---:|---:|---:|---:|---|
| Base, no RAG | | | | | |
| Base + RAG | | | | | |
| LoRA, no RAG | | | | | |
| LoRA + RAG | | | | | |
| QLoRA + RAG | | | | | |

RAG supplies user knowledge at inference time; fine-tuning changes model behavior/task performance. They solve different problems. Record hardware, training time, model ID, package versions, and hyperparameters.
