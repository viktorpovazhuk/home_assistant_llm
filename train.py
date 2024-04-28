import os 
import warnings
from pathlib import Path
import argparse
import contextlib
from dataclasses import dataclass, field

import pandas as pd

import wandb
import torch
import transformers
import bitsandbytes
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (AutoModelForCausalLM,
                         AutoTokenizer,
                         BitsAndBytesConfig,
                         TrainingArguments,
                         pipeline,
                         logging,
                         TrainerCallback,
                         HfArgumentParser)

from peft import PeftModel

@dataclass
class MyTrainingArguments:
    seq_max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    hf_key: str
    wandb_key: str
    hf_model_name: str
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    resume: bool
    adapter_model: str = field(default=None)
    run_name: str

@dataclass
class DataArguments:
    data_path: str
    output_path: str

def generate_prompt(datapoint, tokenizer):
    inp = tokenizer.apply_chat_template([{'role': 'user', 'content': datapoint['user']}, {'role': 'assistant', 'content': datapoint['assistant']}], tokenize=False) + '<eos>'
    return inp

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

def main():
    parser = transformers.HfArgumentParser(
        (DataArguments, MyTrainingArguments)
    )
    (
        data_args, training_args
    ) = parser.parse_args_into_dataclasses()

    data_dir = Path(data_args.data_path)
    output_dir = Path(data_args.output_path)

    hf_key = training_args.hf_key
    wandb_key = training_args.wandb_key

    hf_model_name = training_args.hf_model_name

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_df = pd.read_csv(data_dir / 'home-assistant/train.csv')
    val_df = pd.read_csv(data_dir / 'home-assistant/val.csv')

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name,
                                            trust_remote_code=True,
                                            token=hf_key)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'

    train_df = pd.DataFrame(train_df.apply(generate_prompt, axis=1, tokenizer=tokenizer), columns=['text'])
    val_df = pd.DataFrame(val_df.apply(generate_prompt, axis=1, tokenizer=tokenizer), columns=['text'])
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)

    response_template_with_context = "<start_of_turn>model"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )
    with contextlib.redirect_stdout(None):
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            device_map='auto',
            quantization_config=bnb_config,
            token=hf_key
        )

    if not training_args.resume:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"]
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, peft_config)
    else:
        peft_config = PeftConfig.from_pretrained(training_args.adapter_model)
        model = PeftModel.from_pretrained(model, training_args.adapter_model)

    model.print_trainable_parameters()

    run_name = training_args.run_name
    output_dir = str(output_dir / run_name)

    callbacks = [PeftSavingCallback]
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        logging_dir="logs",
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=3,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_strategy='steps',
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=training_args.eval_steps,
        do_eval=True,
        run_name=run_name,
        disable_tqdm=False
    )
    trainer = SFTTrainer(
        data_collator=collator,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        callbacks=callbacks,
        max_seq_length=training_args.seq_max_length
    )
    model.config.use_cache = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wandb.login(key=wandb_key)

    trainer.train()

    wandb.finish()

if __name__ == '__main__':
    main()