"""Generate a small qualitative evaluation file for base model or a LoRA adapter."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter",default=None)
    p.add_argument("--questions",default="data/training/eval_questions.json")
    p.add_argument("--output",default="outputs/evaluation.json")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16 if device=="cuda" else torch.float32,device_map="auto" if device=="cuda" else None)
    if a.adapter: model=PeftModel.from_pretrained(model,a.adapter)
    if device=="cpu": model.to(device)
    results=[]
    for q in json.loads(Path(a.questions).read_text(encoding="utf-8")):
        prompt=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
        inputs=tok(prompt,return_tensors="pt"); inputs={k:v.to(model.device) for k,v in inputs.items()}
        with torch.no_grad(): out=model.generate(**inputs,max_new_tokens=220,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ans=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        results.append({"question":q,"answer":ans}); print("\nQ:",q,"\nA:",ans)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Saved",a.output)
if __name__=="__main__": main()
