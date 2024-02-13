# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.embeddings import resolve_embed_model
from llama_index.llms import Ollama
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data/text").load_data()

    # bge-m3 embedding model
    embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama
    llm = Ollama(model="mistral", request_timeout=60.0)

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

print(response)
