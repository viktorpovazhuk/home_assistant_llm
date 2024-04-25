import os 
import warnings
from pathlib import Path
import argparse

import pandas as pd

import wandb
import torch
import transformers
import bitsandbytes
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                         AutoTokenizer,
                         BitsAndBytesConfig,
                         TrainingArguments,
                         pipeline,
                         logging,
                         TrainerCallback)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def generate_prompt(datapoint, tokenizer):
    inp = tokenizer.apply_chat_template([{'role': 'user', 'content': datapoint['user']}, {'role': 'assistant', 'content': datapoint['assistant']}], tokenize=False)
    return inp

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--hf_key', type=str)
    parser.add_argument('--wandb_key', type=str)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    hf_key = args.hf_key
    wandb_key = args.wandb_key

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_df = pd.read_csv(data_dir / 'home-assistant/train.csv')
    val_df = pd.read_csv(data_dir / 'home-assistant/val.csv')

    hf_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, 
                                            trust_remote_code=True,
                                            token=hf_key)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'

    train_df = pd.DataFrame(train_df.apply(generate_prompt, axis=1, tokenizer=tokenizer), columns=['text'])
    val_df = pd.DataFrame(val_df.apply(generate_prompt, axis=1, tokenizer=tokenizer), columns=['text'])
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = False,
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_compute_dtype = compute_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        device_map='auto',
        quantization_config=bnb_config,
        token=hf_key
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    project = "home-assistant"
    base_model_name = "mistral-7b-instruct"
    run_name = base_model_name + "-" + project
    output_dir = str(output_dir / run_name)

    callbacks = [PeftSavingCallback]
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        logging_dir="logs",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_strategy='steps',
        logging_steps=25,
        save_steps=50,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio = 0.05,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb",
        evaluation_strategy="epoch",
        do_eval=True,
        run_name=run_name,
        disable_tqdm=False
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        callbacks=callbacks,
        packing=False,
        max_seq_length=512
    )
    model.config.use_cache = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wandb.login(key=wandb_key)

    trainer.train()

    wandb.finish()

if __name__ == '__main__':
    main()