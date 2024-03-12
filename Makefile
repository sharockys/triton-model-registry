run: 
	tritonserver --model-repository=/workspaces/triton-model-registry/model_repository --model-control-mode=poll

query-tokenizer:
	curl -X POST http://localhost:8000/v2/models/TOKENIZER-multilingual-e5-large-instruct/infer -d '{"inputs": [{"name": "INPUT_TEXT", "datatype": "BYTES", "shape": [2, 1], "data": ["Hello, Triton!", "Hello, Triton2!"]}]}'

query-onnx:
	curl -X POST http://localhost:8000/v2/models/ONNX-embedding-multilingual-e5-large-instruct/infer -d '{"inputs": [{"name": "input_ids", "datatype": "INT64", "shape": [2, 4], "data": [[1, 2, 3, 4], [1 ,2 ,3, 4]]}, {"name": "attention_mask", "datatype": "INT64", "shape": [2, 4], "data": [[1,1,1,1], [1,1,1,1]]} ]}'

query-ensemble:
	curl -X POST http://localhost:8000/v2/models/ENSEMBLE-embedding-multilingual-e5-large-instruct/infer -d '{"inputs": [{"name": "INPUT_TEXT", "datatype": "BYTES", "shape": [2, 1], "data": ["Hello, Triton!", "Hello, Triton2!"]}]}'