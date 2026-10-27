import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant.config import MAX_NEW_TOKENS

class RAGAssistant:
    def __init__(self, model_name, embedding_name):
        self.model_name, self.embedding_name = model_name, embedding_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_name, device=self.device)
        self.chunks, self.embeddings = [], None
        self.model = self.tokenizer = None

    def index_documents(self, chunks):
        new_chunks = list(chunks)
        new_embeddings = None if not new_chunks else self.embedder.encode(
            [c.text for c in new_chunks], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.chunks = new_chunks
        self.embeddings = new_embeddings

    def retrieve(self, query, top_k=4):
        if self.embeddings is None:
            return []
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        scores = self.embeddings @ q
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in np.argsort(scores)[::-1][:top_k]]

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.device == "cpu":
                self.model.to(self.device)
        return self.model, self.tokenizer

    def answer(self, question, top_k=4):
        found = self.retrieve(question, top_k)
        if found:
            context = "\n\n".join(f"[Source {i+1}: {d.source}]\n{d.text}" for i, (d, _) in enumerate(found))
            user_text = ("Use the context to answer. If the answer is not in the context, say you could not find it in the provided documents. Do not invent facts.\n\n"
                         f"CONTEXT:\n{context}\n\nQUESTION: {question}")
        else:
            user_text = ("No personal documents are indexed. Answer generally and state uncertainty when appropriate.\n\nQUESTION: " + question)
        model, tok = self.load_model()
        messages = [{"role":"system","content":"You are a helpful, careful personalized AI assistant."},
                    {"role":"user","content":user_text}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.08,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        answer = tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        sources = [{"source":d.source,"excerpt":d.text[:220] + ("..." if len(d.text)>220 else "")} for d,_ in found]
        return answer or "I could not generate an answer. Please try again.", sources
