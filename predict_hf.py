from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
from llama_index.core.node_parser.text.sentence import SentenceSplitter

from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from pathlib import Path
import random
import json
import shutil
import sys

from evaluate import evaluate
import argparse

GENERATE_METHODS_DIR = Path('data/docs/manual')
METHODS_DIR = Path('data/docs/methods')
PROMPT_SEEDS_DIR = Path('data/prompts/generation/components')
PROMPT_COMPONENTS_DIR = Path('data/prompts/generation/components')
VAL_PROMPT_COMPONENTS_DIR = Path('data/prompts/validation/components')
GEN_PROMPTS_DIR = Path('data/prompts/generation/output')
VAL_PROMPTS_DIR = Path('data/prompts/validation/output')
PERSIST_DIR = Path("data/persist_dir")
OUTPUT_DIR = Path("output/")
MODELS_PATH = Path('models/')
DATA_DIR = Path('data/')

Settings.embed_model = resolve_embed_model("local:BAAI/bge-base-en-v1.5")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

# load index
index = load_index_from_storage(storage_context, show_progress=True)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

def get_base_prompt_variables():
    json_scheme_prompt = {
        "method": {
            "type": "string"
        },
        "params": {
            "type": "object"
        }
    }

    example_1_json = {
      "method":"Cover.Open",
      "params":
      {
        "id":165
      }
    }

    example_2_json = {
      "method":"Light.Set",
      "params":
      {
        "id":189,
        "on":True,
        "toggle_after":30,
      }
    }

    with open(VAL_PROMPT_COMPONENTS_DIR / 'instruction.md') as f:
      instruction = f.read()

    variables = {
    "instruction": instruction,
    "json_scheme": "The output JSON should follow the next scheme: " + json.dumps(json_scheme_prompt),
    "example_1": """Devices: Entryway Smoke 36 id=229, Attic Cover 23 id=165, Kitchen Temperature 55 id=257
Methods:
API method 1:
Method name: Cover.Open
Method description: 
Properties:
{"id": {"type": "number", "description": "The numeric ID of the Cover component instance"}, "duration": {"type": "number", "description": "If duration is not provided, Cover will fully open, unless it times out because of maxtime_open first. If duration (seconds) is provided, Cover will move in the open direction for the specified time. duration must be in the range [0.1..maxtime_open]Optional"}}
Response:
null on success; error if the request can not be executed or failed

Command: Open the Attic Cover 23.
JSON: """ + json.dumps(example_1_json),

    "example_2": """Devices: Garage Cover 12 id=175, Study room Light 13 id=189, Bedroom Switch 7 id=335, Bedroom Smoke 10 id=187, Greenhouse Temperature 9 id=457, Living room Humidity 9 id=138
Methods:
API method 1:
Method name: Light.Set
Method description: This method sets the output and brightness level of the Light component. It can be used to trigger webhooks. More information about the events triggering webhooks available for this component can be found below.
Request
Parameters:
{"id": {"type": "number", "description": "Id of the Light component instance. Required"}, "on": {"type": "boolean", "description": "True for light on, false otherwise. Optional"}, "brightness": {"type": "number", "description": "Brightness level Optional"}, "transition_duration": {"type": "number", "description": "Transition time in seconds - time between change from current brightness level to desired brightness level in request Optional"}, "toggle_after": {"type": "number", "description": "Optional flip-back timer in seconds. Optional"}}

Command: Turn on Study room Light 13. And automatically turn it off after half a minute.
JSON: """ + json.dumps(example_2_json),
    }

    return variables
def get_base_prompt():
    base_prompt_template = """
{instruction}
{json_scheme}

{example_1}
    """

    variables = get_base_prompt_variables()

    base_prompt = base_prompt_template.format(**variables)

    return base_prompt

# {example_1}

# {example_2}

def get_user_prompt_template():
    user_prompt_template = """Devices: {env}
Methods:
{methods_description}

Command: {user_cmd}
JSON:
    """

    return user_prompt_template

def predict_prompt(prompt):
    chat = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt').to("cuda")
    prompt_tokens_len = len(prompt[0])
    response = llm.generate(input_ids=prompt, max_new_tokens=4000-prompt_tokens_len)

    response = tokenizer.decode(response[0][prompt_tokens_len:], skip_special_tokens=True)
    if '```json' in response:
        start_i = response.find('```json') + 7
        response = response[start_i:]
    if '```' in response:
        stop_i = response.find('```')
        response = response[:stop_i]
    response = response.strip('\n')

    try:
        json_cmd = json.dumps(json.loads(response))
    except Exception as e:
        print(e)
        print(response)
        return ""

    return json_cmd

def get_methods_description(retrieved_nodes):
    methods_names = []
    methods_description = ''
    for k, node in enumerate(retrieved_nodes):
        methods_description += f'API method {k}:\n{node.text}\n\n'
        method_name = node.metadata['file_name'].replace('.md', '')
        methods_names.append(method_name)

    methods_names = ','.join(methods_names)
    methods_description = methods_description.strip('\n')

    return methods_names, methods_description

def predict(df, run_name, num_nodes=3, selected_devices=None, selected_ids=None, limit_rows=None, verbose=False):
    output_path = OUTPUT_DIR / run_name / 'output.csv'

    if selected_devices:
        df = df[df['device'].isin(selected_devices)].sort_index()
    
    if selected_ids:
        df = df[df['id'].isin(selected_ids)]

    if limit_rows:
        df = df.iloc[:limit_rows]

    output_df = pd.DataFrame(columns=['id', 'mtd', 'json_cmd'])
    for i, row in df.iterrows():
        print(i)

        num_nodes = 3
        
        user_cmd = row['user_cmd']

        env = row['env']

        retrieval_prompt = "Represent this sentence for searching relevant passages: " + user_cmd
        retrieved_nodes = retriever.retrieve(retrieval_prompt)
        
        completed = False
        while (not completed) and (num_nodes > 0):
            try:
                methods_names, methods_description = get_methods_description(retrieved_nodes[:num_nodes])

                user_prompt = get_user_prompt_template().format(**{'env': env, 
                                                            'methods_description': methods_description, 
                                                            'user_cmd': user_cmd})
                prompt = get_base_prompt() + '\n\n' + user_prompt

                json_cmd = predict_prompt(prompt)

                completed = True
            except Exception as e:
                print(e)

                num_nodes -= 1
        
        if verbose:
            print(f'{prompt}\n')
            print('<<<------------------------------------------>>>\n\n')
        
        if json_cmd == "":
            continue

        output_series = pd.Series({'id': row['id'], 'mtd': methods_names, 'json_cmd': json_cmd})
        output_df.loc[len(output_df)] = output_series

        if i == df.index[0]:
            header = True
            mode = 'w'
        else:
            header = False
            mode = 'a'
        output_df.iloc[[len(output_df)-1]].to_csv(output_path, index=False, header=header, mode=mode)

        if i % 300 == 0:
            sys.stdout.flush()
    
    return output_df

# # # # # # # # # # # # 

GT_PATH = DATA_DIR / 'datasets/merged/test_0.csv'
RUN_NAME = 'gemma_2b_it_hf'
NUM_EXAMPLES = 2
NUM_NODES = 3
MODEL_NAME = 'google/gemma-1.1-2b-it'
N_CTX = 4000
settings = {
    'llm': MODEL_NAME,
    'num_examples': NUM_EXAMPLES,
    'num_nodes': NUM_NODES,
    'n_ctx': N_CTX
}

# # # # # # # # # # # # 

(OUTPUT_DIR / RUN_NAME).mkdir(exist_ok=True)

gt_df = pd.read_csv(GT_PATH)

with open(OUTPUT_DIR / RUN_NAME / "settings.json", 'w') as f:
    f.write(json.dumps(settings))
shutil.copy(VAL_PROMPT_COMPONENTS_DIR / 'instruction.md', OUTPUT_DIR / RUN_NAME)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype=torch.float16,
    quantization_config=nf4_config
)

output_df = predict( gt_df, RUN_NAME, num_nodes=NUM_NODES, verbose=False)

json_schemes_df = pd.read_csv(METHODS_DIR.parent / 'methods_json.csv')

output_df = pd.read_csv(OUTPUT_DIR / RUN_NAME / 'output.csv')

with open(OUTPUT_DIR / RUN_NAME / 'settings.json') as f:
    settings = f.read()
settings = json.loads(settings)

evaluate(gt_df, output_df, json_schemes_df, RUN_NAME, settings, OUTPUT_DIR, save_intermediate=True)