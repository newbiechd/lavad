from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen3-14B', cache_dir='./libs/', revision='master')