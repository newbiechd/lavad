from modelscope import snapshot_download

model_dir = snapshot_download('LLM-Research/llama-2-7b-chat', cache_dir='./libs/', revision='master')