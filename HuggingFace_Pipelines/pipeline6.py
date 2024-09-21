from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = """Remo visited Paysandu this Saturday and won the Para classic game 3-2, in
an electrifying duel for the 9th round of the Brazilian Championship Series C.
In a lively second half, Leao opened the score 2-0 with Helio and Marlon, but
saw Papao react and tie in the final minutes, with Wesley Matos and Nicolas.
At 43, Wallace scored the third and sealed the victory. Remo won the classic game
against Paysandu in Series C. With the result, Remo took the provisional lead
of group A with 16 points, against 15 for Santa Cruz, who still plays in the
round and can retake the lead. Paysandu, on the other hand, remained at 11 points,
in 5th place, outside the zone of qualifying for the quarterfinals.
"""

summary = summarizer(text, max_length=20, min_length=5, do_sample=False)

print(summary)