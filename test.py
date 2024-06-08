import ChatTTS
from IPython.display import Audio
import os
import time
import requests
import openai
import torchaudio
import torch
import numpy as np

ollama_api_key = 'llama3'
ollama_api_url = 'http://localhost:11434/v1'
ollama_client = openai.OpenAI(api_key=ollama_api_key, base_url=ollama_api_url)

def ollama_chat(user_query):
    response = ollama_client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": """你是个好人"""},
            {"role": "user", "content": user_query}
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    return response.choices[0].message.content

chat = ChatTTS.Chat()
chat.load_models(compile=False) # 设置为True可获得更好的性能

say = ollama_chat("只要问我一个问题")

print(say)

texts = [say,]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)