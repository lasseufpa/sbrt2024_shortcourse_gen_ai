from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I will not say anything")

#, 

print(res)