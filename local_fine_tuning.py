#!/usr/bin/env python
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer

model_checkpoint = 'distilbert/distilgpt2'
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, use_fast=True)

dataset = load_dataset("rajpurkar/squad")

# Remove useless columns
dataset = dataset.remove_columns(["id", "title"])
# Reduce the dataset
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(10000))
dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(2000))

special_tokens = tokenizer.special_tokens_map

# create tokenize function
def tokenize_function(input_data):
    predict = f'{input_data["context"]} Q:{input_data["question"]} A:{input_data["answers"]["text"][0]}{special_tokens["bos_token"]}'
    inputs = tokenizer(
        predict,
        truncation=True,
    )
    return inputs

dataset = dataset.map(tokenize_function, remove_columns=['context', 'question', 'answers'])

# Reallocate tokens to fixed block size
def reallocate_tokens(input_data, max_token_size=128):
    combined_tokens = {
        key: sum(values, [])
        for key, values in input_data.items()
    }

    total_length = len(combined_tokens[list(input_data.keys())[0]])
    total_length = (total_length // max_token_size) * max_token_size

    token_blocks = {
        key: [tokens[i: i + max_token_size] for i in range(0, total_length, max_token_size)]
        for key, tokens in combined_tokens.items()
    }

    token_blocks['labels'] = token_blocks['input_ids'].copy()
    return token_blocks

dataset = dataset.map(reallocate_tokens,batched=True)

train_dataset = dataset['train']
eval_dataset = dataset['validation']

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

training_args = TrainingArguments(
    f'./{model_checkpoint}-ft',
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()