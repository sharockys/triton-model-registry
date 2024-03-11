# Deploy Huggingface embedding model to Triton Inference Server   
I created this repo to facilitate my own work with deploying Huggingface models to Triton Inference Server. In my homelab case, I use this to create embeddings for my Qdrant embedding index. 

## Export the model from Hugging Face to ONNX
Install the `optimum-cli` tool from the [Optimum](https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model) 
```bash
pip install optimum[exporters]
```

```bash
optimum-cli export onnx --model intfloat/multilingual-e5-large-instruct multilingual-e5-large-instruct
```

Then put the tokenizer in the directory with Python backend and the ONNX model in the directory with the ONNX backend like: 
```txt
model_repository
├── ONNX-embedding-multilingual-e5-large-instruct
│   ├── 1
│   │   ├── config.json
│   │   ├── model.onnx
│   │   └── model.onnx_data
│   └── config.pbtxt
└── TOKENIZER-multilingual-e5-large-instruct
    ├── 1
    │   ├── model.py
    │   └── tokenizer
    │       ├── sentencepiece.bpe.model
    │       ├── special_tokens_map.json
    │       ├── tokenizer.json
    │       └── tokenizer_config.json
    └── config.pbtxt
```

## dev container
You can use devcontainer in VSCode to develop and test the model. 
Open the command palette and select `Dev Container: Reopen in Container`.  
Then 
```bash
make run 
```

## query the model
You can query the model in the dev container or use port-forwarding to query the model from your local machine. 
```bash
make query-tokenizer 
make query-onnx 
```
Example data is included by the makefile.

## Deploy the model to Triton Inference Server
Checkout [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) for more information. 

Me personally use the S3 and make triton to pull the model from the S3. All you have to do is add the environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` to the triton container and add the argument for s3 bucket.
```bash
$ tritonserver --model-repository=s3://bucket/path/to/model/repository ...
```


## TODOs
- [ ] Perf Analyzer guide 
- [ ] Add ensemble model with average pooling
- [ ] More backends
- [ ] Optimize the inference for RTX 3090 