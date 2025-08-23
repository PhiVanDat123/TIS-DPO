from datasets import load_dataset

dataset = load_dataset(
    "json", 
    data_files={
        "train": "datasets/ultra-feedback/train.jsonl",
        "test": "datasets/ultra-feedback/test.jsonl"
    }
)

print(dataset.column_names)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token  # nếu model không có pad token

def tokenize_chosen_rejected(batch):
    # Tokenize 'chosen'
    chosen_tokens = tokenizer(
        batch['chosen'], 
        padding='max_length', 
        truncation=True, 
        max_length=256
    )
    
    # Tokenize 'rejected'
    rejected_tokens = tokenizer(
        batch['rejected'], 
        padding='max_length', 
        truncation=True, 
        max_length=256
    )

    return {
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_tokens["input_ids"],

        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        "rejected_labels": rejected_tokens["input_ids"],
    }

tokenized_dataset = dataset.map(
    tokenize_chosen_rejected, 
    batched=True, 
    batch_size=32, 
    num_proc=8  # số process parallel, nếu server có nhiều CPU core
)

print(tokenized_dataset.column_names)

tokenized_dataset["train"].to_json("datasets/ultra-feedback-tokenized/train.jsonl")
tokenized_dataset["test"].to_json("datasets/ultra-feedback-tokenized/test.jsonl")
