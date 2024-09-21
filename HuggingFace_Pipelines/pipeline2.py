from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator("I am a bad person and I will", max_length=300, num_return_sequences=1)

print(res)