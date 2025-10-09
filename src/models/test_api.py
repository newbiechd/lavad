import openai

client = openai.OpenAI(
    api_key="sk-AMy9-RdL55xB30zIVE94kA",
    base_url="https://llmapi.paratera.com/v1"
)

response = client.chat.completions.create(
    model="DeepSeek-R1",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)
print(response)