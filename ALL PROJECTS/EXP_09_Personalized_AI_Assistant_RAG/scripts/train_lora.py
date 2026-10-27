"""LoRA/optional QLoRA starter. JSONL rows need instruction, optional input, and output."""
import argparse, json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data",default="data/training/train.jsonl")
    p.add_argument("--output",default="outputs/lora-adapter")
    p.add_argument("--epochs",type=float,default=1)
    p.add_argument("--max-length",type=int,default=512)
    p.add_argument("--qlora",action="store_true")
    a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    kwargs={}
    if a.qlora:
        if not torch.cuda.is_available(): raise RuntimeError("QLoRA requires a compatible CUDA GPU and bitsandbytes. Use standard LoRA otherwise.")
        from transformers import BitsAndBytesConfig
        kwargs={"quantization_config":BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True),"device_map":"auto"}
    else:
        kwargs["torch_dtype"]=torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available(): kwargs["device_map"]="auto"
    model=AutoModelForCausalLM.from_pretrained(a.model,**kwargs)
    from peft import LoraConfig, get_peft_model
    if a.qlora:
        from peft import prepare_model_for_kbit_training
        model=prepare_model_for_kbit_training(model)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()
    rows=[]
    for line in Path(a.data).read_text(encoding="utf-8").splitlines():
        if line.strip():
            x=json.loads(line)
            rows.append({"text":f"### Instruction:\n{x.get('instruction','')}\n### Input:\n{x.get('input','')}\n### Response:\n{x.get('output','')}"})
    if not rows: raise ValueError("No examples found in training JSONL.")
    ds=Dataset.from_list(rows)
    ds=ds.map(lambda b:tokenizer(b["text"],truncation=True,max_length=a.max_length),batched=True,remove_columns=["text"])
    args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,per_device_train_batch_size=1,
        gradient_accumulation_steps=8,learning_rate=2e-4,logging_steps=1,save_strategy="epoch",
        report_to="none",fp16=torch.cuda.is_available() and not a.qlora,gradient_checkpointing=True,
        optim="paged_adamw_8bit" if a.qlora else "adamw_torch")
    trainer=Trainer(model=model,args=args,train_dataset=ds,data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False))
    trainer.train()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    model.save_pretrained(a.output); tokenizer.save_pretrained(a.output)
    print("Saved adapter to",a.output)
if __name__=="__main__": main()
