# Personalized AI Assistant using Open-Source LLM, RAG, and LoRA/QLoRA

A research starter project for building a personalized AI assistant that answers questions from user-specific documents. It combines an open-source instruction model, semantic retrieval, and optional parameter-efficient fine-tuning.

## Aim
Develop a personalized AI assistant using an open-source foundational language model, integrate RAG to answer questions from user-specific documents, and investigate how LoRA or QLoRA affects response quality.

## Features
- Streamlit chat UI.
- Default Hugging Face model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Upload PDF, DOCX, TXT, and Markdown files.
- Sentence Transformers embeddings and cosine-similarity retrieval.
- Relevant document passages and source excerpts shown with answers.
- LoRA/optional QLoRA training starter.
- Evaluation helper and experiment rubric.

## Structure
```text
personalized_ai_assistant_project/
├── app.py
├── assistant/ (config.py, documents.py, rag.py)
├── scripts/ (index_folder.py, train_lora.py, evaluate.py)
├── data/
│   ├── knowledge_base/
│   ├── training/ (train.jsonl, eval_questions.json)
│   └── vector_store/
├── docs/EXPERIMENTS.md
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements
Python 3.10 or 3.11 recommended, internet for first model download, and preferably 16 GB RAM for CPU inference. CPU is supported but can be slow. A compatible NVIDIA GPU is recommended for training. QLoRA requires a compatible CUDA environment and `bitsandbytes`; support varies by OS.

## Setup on Windows / VS Code
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Use the project virtual environment so installed package versions match `requirements.txt`.

## Run the app
From this project folder, start the Flask web app:
```powershell
python run.py
```
Open `http://127.0.0.1:5009`. Upload files in the left panel, click **Index Documents**, and then ask questions in the chat. The first answer may take longer while the embedding and language models load. Uploaded files are indexed in memory, so re-upload them after restarting.

To use the alternate Streamlit interface instead:
```powershell
python run.py --ui streamlit
```
Open the local URL it prints, usually `http://127.0.0.1:8509`. Do not run `streamlit run app.py`: `app.py` is the Flask backend; the Streamlit interface is `streamlit_app.py`.

To embed files in `data/knowledge_base/` and export a local index:
```powershell
python scripts/index_folder.py
```
(The current Streamlit app uses session-memory indexing; folder indexing is a separate utility.)

## LoRA fine-tuning
The included dataset is a tiny format demonstration, not a valid research-sized training set. Expand it with relevant, permitted examples in `data/training/train.jsonl`, one JSON object per line:
```json
{"instruction":"Explain RAG simply.","input":"","output":"RAG retrieves relevant information before generating an answer."}
```
Run a small LoRA experiment:
```powershell
python scripts/train_lora.py --epochs 1
```
Use a supported GPU if possible; CPU training may be extremely slow. Optional QLoRA:
```bash
python scripts/train_lora.py --qlora --epochs 1
```
QLoRA requires a compatible CUDA GPU and bitsandbytes; it is not recommended for CPU-only Windows.

## Evaluation
```powershell
python scripts/evaluate.py --output outputs/base_answers.json
python scripts/evaluate.py --adapter outputs/lora-adapter --output outputs/lora_answers.json
```
Compare outputs using the rubric in `docs/EXPERIMENTS.md`. The script generates answers; it does not automatically score them.

## How it works
1. Extract text from user documents.
2. Split text into overlapping chunks.
3. Embed chunks into vectors.
4. Retrieve passages most similar to the question.
5. Add retrieved passages to the LLM prompt.
6. Generate an answer and display source excerpts.

## Limitations and responsible use
- First run downloads model and embedding weights.
- Small models can make mistakes; verify important answers.
- Training data included here is illustrative only and is insufficient to prove quality improvements.
- Do not commit private documents, secrets, model caches, or personal data to a public repository.
- Check the license and usage terms of the chosen model and dataset.

## Author
Prachi Shinde

## License
For educational and research use. Check pretrained model and dataset licenses before redistribution or deployment.
