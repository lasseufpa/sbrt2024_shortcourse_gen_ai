from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier("This is my candidate: I want my university degree", 
	candidate_labels=["education", "politics", "nerd"])

print(res)