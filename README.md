# LLM-based Voice Control System for Smart Device Environment

This repo is an official code of **LLM-based Voice Control System for Smart Device Environment** paper.

## Usage

### Environment setup

```
conda create -y --name assistant python=3.10.13=h955ad1f_0 --file requirements.txt
```

### Dataset preparation

Download data folder: [link](https://drive.google.com/drive/folders/1OIOUpLZ-OyfNY7mwQcIg0cCLlmtCvccq?usp=sharing). And put it into the project root folder.

### Weights

#### Pretrained models

| Name of base model  | Weights |
| ------------- | ------------- |
| Mistral-7B GGUF  | [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf?download=true)  |
| Gemma-2B  | [Link](https://huggingface.co/google/gemma-1.1-2b-it)  |
| Mistral-7B  | [Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  |

#### Fine-tuned adapters

| Name of adapter base model  | Adapter weights |
| ------------- | ------------- |
| Gemma-2B  | [Link](https://huggingface.co/viktorpovazhuk/gemma-1.1-2b-it-vcs)  |
| Mistral-7B  | [Link](https://huggingface.co/viktorpovazhuk/mistral-7b-instruct-v0.2-vcs)  |

### Training

To train QLoRA adapter for Gemma-2B.

```
python train.py --data_path data/datasets/merged/home-assistant \
    --output_path output \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --seq_max_length 4100 \
    --hf_key YOUR_KEY \
    --wandb_key YOUR_KEY
```

To train QLoRA adapter for Mistral-7B.

```
python train_mistral.py --data_path data/datasets/merged/home-assistant \
    --output_path output \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --seq_max_length 4100 \
    --hf_key YOUR_KEY \
    --wandb_key YOUR_KEY
```

### Prediction and evaluation

To predict with pretrained Mistral-7B download corresponding GGUF model from [Pretrained models](#pretrained-models) and put it in ```models``` folder. Set the RUN_NAME in ```predict.py```. You can also play with NUM_EXAMPLES and NUM_NODES. Then run:

```
python predict.py
```

To predict for fine-tuned Gemma-2B or Mistral-7B download corresponding adapter weights from [Fine-tuned adapters](#fine-tuned-adapters) and put it in ```models``` folder. You may also download pretrained base model, or let it be downloaded automatically. Then run:

```
python predict_hf_lora.py --run_name gemma-2b-run \
    --model_name google/gemma-1.1-2b-it \
    --peft_model models/gemma-1.1-2b-it-vcs
```