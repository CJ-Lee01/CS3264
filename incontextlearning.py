import sys


from dataloading import get_hasib18_fns

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

if __name__ == "__main__":
    pass

# Load your dataset
df, _ = get_hasib18_fns()  # Must contain 'text' and 'label' columns
df = df.dropna(subset=["text", "label"]).sample(10_000, random_state=1)

df0 = df[df["label"] == 0][:1] # Real
df1 = df[df["label"] == 1][:1] # Fake

part2 = "\n".join([f"Article: {text} \nLabel: Real" for text, label in zip(df0["text"], df0["label"])])
part1 = "\n".join([f"Article: {text} \nLabel: Fake" for text, label in zip(df1["text"], df1["label"])])

# Load model & tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.eval()




# Few-shot prompt template
few_shot = f"""
You are a fake news detector. Classify each article as either "Real" or "Fake", and explain your decision.
"""


def classify_article(article):
    prompt = few_shot + f"Article: {article}\n\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the label prediction
    print(f"Response: {decoded}\n\n\n")
    if "Fake" in decoded:
        return 0
    elif "Real" in decoded:
        return 1
    else:
        print(f"LLM misbehaved with: \ninput {article}\noutput:{decoded}")
        return "Unknown"


# 🔁 Batch inference
predictions = []
counter = 0
correct_pred = 0
for article in tqdm(df["text"].tolist()):
    pred = classify_article(article)
    predictions.append(pred)
    if pred == df["label"].tolist()[counter]:
        correct_pred += 1
    counter += 1
    if counter % 200 == 0:
        print(f"Accuracy: {correct_pred/counter}, {correct_pred}/{counter}")

df["predicted"] = predictions
accuracy = (df["predicted"] == df["label"]).mean()
print(f"Accuracy: {accuracy:.2%}")

# Save results
df.to_csv("predictions_with_llm_w_context.csv", index=False)
