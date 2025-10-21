from datasets import load_dataset
import random
import json
import os

# Load GSM8K dataset
print("Loading GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main")
train_data = dataset["train"]

print(f"Total questions in GSM8K train set: {len(train_data)}")

# Randomly sample 250 questions
sampled_data = random.sample(list(train_data), 250)

# Format the data
formatted_data = [
    {
        "id": i,
        "question": item["question"].strip(),
        "full_answer": item["answer"].strip(),
        "final_answer": item["answer"].strip().split("####")[-1].strip()  # Extract final numeric answer
    }
    for i, item in enumerate(sampled_data)
]

# Save to question-answer pairs to JSON file
mkdir_path = "data"
os.makedirs(mkdir_path, exist_ok=True)
output_path = "data/gsm8k.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(formatted_data)} questions to {output_path}")

# Optional: print first example
print("\nExample:")
print(json.dumps(formatted_data[0], indent=2))
