import os
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

import wandb

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import (
                         TrainingArguments,
                         TrainerCallback,
                         HfArgumentParser)
from unsloth import FastLanguageModel

@dataclass
class MyTrainingArguments:
    seq_max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    hf_key: str
    wandb_key: str

@dataclass
class DataArguments:
    data_path: str
    output_path: str

def generate_prompt(datapoint, tokenizer):
    inp = tokenizer.apply_chat_template([{'role': 'user', 'content': datapoint['user']}, {'role': 'assistant', 'content': datapoint['assistant']}], tokenize=False)
    return inp

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

def main():
    parser = HfArgumentParser(
        (DataArguments, MyTrainingArguments)
    )
    (
        data_args, training_args
    ) = parser.parse_args_into_dataclasses()

    data_dir = Path(data_args.data_path)
    output_dir = Path(data_args.output_path)

    hf_key = training_args.hf_key
    wandb_key = training_args.wandb_key

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_df = pd.read_csv(data_dir / 'home-assistant/train.csv')
    val_df = pd.read_csv(data_dir / 'home-assistant/val.csv')

    # to auto identify
    dtype = None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        max_seq_length=training_args.seq_max_length,
        dtype=dtype,
        load_in_4bit=True,
        token=hf_key
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'   
    
    train_df = pd.DataFrame(train_df.apply(generate_prompt, axis=1, tokenizer=tokenizer), columns=['text'])
    val_df = pd.DataFrame(val_df.apply(generate_prompt, axis=1, tokenizer=tokenizer), columns=['text'])
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none", # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    project = "home-assistant"
    base_model_name = "mistral-7b-instruct"
    run_name = base_model_name + "-" + project
    output_dir = str(output_dir / run_name)

    callbacks = [PeftSavingCallback]
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        logging_dir="logs",
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=3,
        gradient_accumulation_steps=3,
        optim="adamw_8bit",
        save_strategy='steps',
        logging_steps=2,
        save_steps=50,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_steps=5,
        group_by_length=True,
        lr_scheduler_type="cosine",
        # report_to="wandb",
        evaluation_strategy="epoch",
        do_eval=True,
        run_name=run_name,
        disable_tqdm=False
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=training_args.seq_max_length,
        dataset_num_proc=2,
        packing=False,
        args=training_arguments,
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # wandb.login(key=wandb_key)

    trainer.train()

    # wandb.finish()

if __name__ == '__main__':
    main()