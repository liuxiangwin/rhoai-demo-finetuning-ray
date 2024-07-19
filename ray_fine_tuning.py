#!/usr/bin/env python
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os

import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func():
    model_checkpoint = 'distilbert/distilgpt2'
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, use_fast=True)

    dataset = load_dataset("rajpurkar/squad")

    # Remove useless columns
    dataset = dataset.remove_columns(["id", "title"])
    # dataset['train'] = dataset['train'].shuffle(seed=42).select(range(10000))
    # dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(2000))
    dataset['train'] = dataset['train']
    dataset['validation'] = dataset['validation']

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

    # Report Metrics and Checkpoints to Ray Train
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

    # Prepare Transformers Trainer
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    trainer.train()

# Use Minio for model checkpoint
def get_minio_run_config():
    import s3fs
    import pyarrow.fs

    s3_fs = s3fs.S3FileSystem(
        key = os.environ['AWS_ACCESS_KEY_ID'],
        secret = os.environ['AWS_SECRET_ACCESS_KEY'],
        endpoint_url = os.environ['AWS_S3_ENDPOINT'],
    )
    custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))

    run_config = ray.train.RunConfig(storage_path='ray-train', storage_filesystem=custom_fs)
    return run_config


# Define a Ray TorchTrainer to launch `train_func` on all workers
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=3, use_gpu=True),
    run_config=get_minio_run_config(),
)
result: ray.train.Result = ray_trainer.fit()

# Load the trained model
with result.checkpoint.as_directory() as checkpoint_dir:
    checkpoint_path = os.path.join(
        checkpoint_dir,
        ray.train.huggingface.transformers.RayTrainReportCallback.CHECKPOINT_NAME,
    )
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

# In-place testing
model.eval()
max_length = 30
prompt = "There are two pencils in the box. Q: How many pencils? A:"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)