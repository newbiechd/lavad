from vllm import LLM

llm = LLM(model="./libs/Qwen/Qwen3-14B")
outputs = llm.generate("Hello, my name is")

print(outputs)