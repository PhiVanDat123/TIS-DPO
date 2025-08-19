from datasets import load_dataset

dataset = load_dataset("datasets/ultra-feedback", split="train")
print(dataset.column_names)
