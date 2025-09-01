import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv('FASHION_TREND_INTELLIGENCE_TOKEN_READ')
print(f"API Key: {HF_TOKEN}")

client = InferenceClient(token=HF_TOKEN)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3-0324",
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)

print(completion.choices[0].message)


def query_huggingface(model_name, inputs, token=None):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {"inputs": inputs}

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# Utilisation
token = os.getenv("HF_TOKEN")
result = query_huggingface("gpt2", "The future of AI is", token)
print(result)