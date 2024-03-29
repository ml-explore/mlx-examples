# MLX RAG GGUF
Minimal, clean code implementation of RAG with mlx inferencing for GGUF models.

The code here builds on <a href="https://github.com/vegaluisjose/mlx-rag">https://github.com/vegaluisjose/mlx-rag</a>, it has been optimized to support RAG-based inferencing for .gguf models. I am using <a href="https://huggingface.co/BAAI/bge-small-en">BAAI/bge-small-en</a> for the embedding model, <a href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf">TinyLlama-1.1B-Chat-v1.0-GGUF</a> as base model and the custom vector database script for indexing texts in a pdf file. Inference speeds can go up to ~413 tokens/sec for prompts and ~36 tokens/sec for generation on my 8G M2 Air.

## Demo

https://github.com/Jaykef/mlx-rag-gguf/assets/11355002/5fb262a9-81af-4a45-b9bb-37b501ff7936


## Usage
Download Models (you can use hf's snapshot_download but I recommend downloading separately to save time)
- <a href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf">tinyllama-1.1b-chat-v1.0.Q4_0.gguf</a> put it in the tinyllama folder.
- <a href="https://huggingface.co/Jaward/mlx-bge-small-en">mlx-bge-small-en</a> converted mlx format of BAAI/bge-small-en, put it in the mlx-bge-small-en folder.
- <a href="https://huggingface.co/BAAI/bge-small-en/blob/main/model.safetensors">bge-small-en</a> Only need the model.safetensors file, put it in the bge-small-en folder.


Install requirements
```
python3 -m pip install -r requirements.txt
```

Convert pdf into mlx compatible vector database
```
python3 create_vdb.py --pdf mlx_docs.pdf --vdb vdb.npz
```

Query the model
```
python3 rag_vdb.py --question "Teach me the basics of mlx"
```

The files in the repo work as follow:

- <a href="https://github.com/Jaykef/mlx-rag-gguf/blob/main/gguf.py">gguf.py</a>: Has all stubs for loading and inferencing .gguf models.
- <a href="https://github.com/vegaluisjose/mlx-rag/blob/main/vdb.py">vdb.py</a>: Holds logic for creating a vector database from a pdf file and saving it in mlx format (.npz) .
- <a href="https://github.com/Jaykef/mlx-rag-gguf/blob/main/create_vdb.py">create_vdb.py</a>: It inherits from vdb.py and has all arguments used in creating a vector DB from a PDF file in mlx format (.npz).
- <a href="https://github.com/Jaykef/mlx-rag-gguf/blob/main/rag_vdb.py">rag_vdb.py</a>: Retrieves data from vdb used in querying the base model.
- <a href="https://github.com/Jaykef/mlx-rag-gguf/blob/main/model.py">model.py</a>: Houses logic for the base model (with configs), embedding model and transformer encoder.
- <a href="https://github.com/Jaykef/mlx-rag-gguf/blob/main/utils.py">utils.py</a>: Utility function for accessing GGUF tokens.

Queries make use of both .gguf (base model) and .npz (retrieval model) simultaneouly resulting in much higher inferencing speeds.

## License
MIT
