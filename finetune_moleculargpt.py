#Adapted from https://github.com/NYUSHCS/MolecularGPT/blob/main/finetune_moleculargpt.py
#prompter.py, ds_config_zero2.json, and templates/alpaca.json must be present in directory
#these can be obtained from the original MolecularGPT GitHub: https://github.com/NYUSHCS/MolecularGPT

import json, os, sys
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, HfArgumentParser, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from prompter import Prompter

def load_adapter(model, d):
    for n in ("adapter_model.bin", "pytorch_model.bin"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            set_peft_model_state_dict(model, torch.load(p, map_location="cpu"))
            return
    raise FileNotFoundError(d)

#saving adapter weights at each checkpoint
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw):
        if state.is_world_process_zero:
            d = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kw["model"].save_pretrained(d)
            p = os.path.join(d, "pytorch_model.bin")
            if os.path.exists(p): os.remove(p)
        return control

#model, tokenizer, and LoRA configuration
@dataclass
class ModelArgs:
    model_name_or_path: str
    tokenizer_name: Optional[str] = None
    prompt_template_name: str = "alpaca"
    use_fast_tokenizer: bool = False
    load_in_bits: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    target_modules: str = "q_proj,k_proj,v_proj,down_proj,up_proj"
    warmstart_adapter_path: Optional[str] = None

    def __post_init__(self):
        self.target_modules = [x.strip() for x in self.target_modules.split(",") if x.strip()]

#Dataset options
@dataclass
class DataArgs:
    train_files: List[str] = field(default_factory=list)
    validation_files: Optional[List[str]] = None
    train_on_inputs: bool = False
    add_eos_token: bool = False
    block_size: int = 512
    preprocessing_num_workers: Optional[int] = 4
    overwrite_cache: bool = False
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

margs, dargs, targs = HfArgumentParser((ModelArgs, DataArgs, TrainingArguments)).parse_args_into_dataclasses()
set_seed(targs.seed)
prompter = Prompter(margs.prompt_template_name)
tok = AutoTokenizer.from_pretrained(margs.tokenizer_name or margs.model_name_or_path, use_fast=margs.use_fast_tokenizer, padding_side="left")
tok.pad_token_id = 0
block = min(dargs.block_size, tok.model_max_length)

#tokenise prompts
def tok1(s, add_eos=True):
    x = tok(s, truncation=True, max_length=block, padding=False, return_tensors=None)
    if add_eos and x["input_ids"] and x["input_ids"][-1] != tok.eos_token_id and len(x["input_ids"]) < block:
        x["input_ids"].append(tok.eos_token_id)
        x["attention_mask"].append(1)
    x["labels"] = x["input_ids"].copy()
    return x

#convert JSON training set into token IDs and labels 
def map_fn(r):
    full = tok1(prompter.generate_prompt(r["instruction"], r["input"], r["output"]), True)
    if not dargs.train_on_inputs:
        user = tok1(prompter.generate_prompt(r["instruction"], r["input"]), dargs.add_eos_token)
        n = len(user["input_ids"]) - (1 if dargs.add_eos_token else 0)
        full["labels"] = [-100] * n + full["labels"][n:]
    return full

#load and tokenise the training set
cache = os.path.join(targs.output_dir, "dataset_cache")
train = load_dataset("json", data_files=dargs.train_files[0], cache_dir=cache)["train"]
if dargs.max_train_samples is not None: train = train.select(range(min(len(train), dargs.max_train_samples)))
train = train.shuffle(seed=targs.seed).map(map_fn, batched=False, num_proc=dargs.preprocessing_num_workers, remove_columns=list(train.features), load_from_cache_file=not dargs.overwrite_cache)

#load and tokenise validation set
val = None
if dargs.validation_files:
    val = load_dataset("json", data_files=dargs.validation_files[0], cache_dir=cache)["train"]
    if dargs.max_eval_samples is not None: val = val.select(range(min(len(val), dargs.max_eval_samples)))
    val = val.shuffle(seed=targs.seed).map(map_fn, batched=False, num_proc=dargs.preprocessing_num_workers, remove_columns=list(val.features), load_from_cache_file=not dargs.overwrite_cache)

bnb = None
load8 = False
if margs.load_in_bits == 4:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 if targs.bf16 else torch.float16)
elif margs.load_in_bits == 8:
    load8 = True

#load the base model
model = AutoModelForCausalLM.from_pretrained(margs.model_name_or_path, torch_dtype=torch.float16, load_in_8bit=load8, quantization_config=bnb, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})

if len(tok) > model.get_input_embeddings().weight.shape[0]: model.resize_token_embeddings(len(tok))

if margs.load_in_bits == 8:
    model = prepare_model_for_int8_training(model)
elif margs.load_in_bits == 4:
    model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, LoraConfig(r=margs.lora_r, lora_alpha=margs.lora_alpha, target_modules=margs.target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

#optional warm-start from previous checkpoint
if margs.warmstart_adapter_path: load_adapter(model, margs.warmstart_adapter_path)

#build the trainer
trainer = Trainer(model=model, args=targs, train_dataset=train if targs.do_train else None, eval_dataset=val if (targs.do_eval and val is not None) else None, tokenizer=tok, data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True), callbacks=[SavePeftModelCallback()])

# Run training and save checkpoint adapters.
if targs.do_train:
    out = trainer.train()
    model.save_pretrained(targs.output_dir)
    tok.save_pretrained(targs.output_dir)
    trainer.log_metrics("train", out.metrics)
    trainer.save_metrics("train", out.metrics)
    trainer.save_state()

#run evaluation
if targs.do_eval and val is not None:
    m = trainer.evaluate()
    trainer.log_metrics("eval", m)
    trainer.save_metrics("eval", m)