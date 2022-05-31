#!/usr/bin/env python3

import math
import random
import argparse
import os
from functools import partial
from typing import Union

import wandb
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.optimization import get_scheduler
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from tqdm import tqdm

# disable huggingface tokenizer parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if os.getenv('USER') == 'agrawalk':
    data_dir = '/home/users/agrawalk/word-learning'
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/users/agrawalk/transformers_cache'
elif os.getenv('USER') == 'ketanagrawal':
    data_dir = '/Users/ketanagrawal/word-learning'
else:
    data_dir = '.'

# TRAIN_CSV_PATH = os.path.join(data_dir, 'Apr112022_child_adult_reformulations_nonull_train.csv')
# VAL_CSV_PATH = os.path.join(data_dir, 'Apr112022_child_adult_reformulations_nonull_val.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print('using device:', device)

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

def validate_model(model: nn.Module, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], dataloader: DataLoader,
                   metric_prefix: str, num_batches=None):
    model.eval()
    assert(dataloader)
    losses = []
    for step, batch in tqdm(enumerate(dataloader)):
        if num_batches is not None and step == num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(dataloader.batch_size))

    losses = torch.cat(losses)
    losses = losses[:len(dataloader.dataset)]
    loss = torch.mean(losses)
    metrics = {}
    try:
        metrics[f'{metric_prefix}/loss'] = loss.item()
        metrics[f'{metric_prefix}/perplexity'] = math.exp(loss)
    except OverflowError:
        metrics[f'{metric_prefix}/perplexity'] = float("inf")
    return metrics

def tokenize_function(examples, input_column, tokenizer):
    return tokenizer(examples[input_column], return_special_tokens_mask=True)

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def run_experiment(config: dict):
    datasets = load_dataset('csv', data_files={
        'train': config['train_csv_path'],
        'val': config['val_csv_path'],
    })
    assert(isinstance(datasets, DatasetDict))
    print(f'Size of training set: {len(datasets["train"])}')
    print(f'Size of validation set: {len(datasets["val"])}')
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'], use_fast=True)
    tokenize_fn = partial(tokenize_function,
                          input_column=config['input_column'],
                          tokenizer=tokenizer)
    tokenized_datasets = datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=[config['input_column']]) # type: ignore

    group_texts_fn = partial(group_texts, block_size=config['block_size'])
    lm_datasets = tokenized_datasets.map(
        group_texts_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    model = AutoModelForMaskedLM.from_pretrained(config['pretrained_model'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm_probability=config['mlm_probability'])
    num_workers = 0

    # Conditional for small test subsets
    if len(lm_datasets['train']) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(lm_datasets['train'])), 3):
            print(f"Sample {index} of the training set: {lm_datasets['train'][index]}.")

    train_dataloader = DataLoader(lm_datasets['train'], shuffle=True, batch_size=config['batch_size'],
                                  collate_fn=data_collator, num_workers=num_workers)
    val_dataloader = DataLoader(lm_datasets['val'], batch_size=config['val_batch_size'],
                                collate_fn=data_collator, num_workers=num_workers)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])


    num_training_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    if config['wandb']:
        wandb.watch(model)
    model.to(device)
    print('Evaluating on validation set...')
    val_metrics = validate_model(model, tokenizer, val_dataloader, 'val/')
    if config['wandb']:
        wandb.log({'train/epoch': 0, **val_metrics})
    else:
        print(f'Val metrics @ epoch 0: {val_metrics}')
    print('Starting training.')
    n_steps_per_epoch = len(train_dataloader)
    for epoch in range(config['num_epochs']):
        print(f'Training epoch {epoch + 1}...')
        model.train()
        train_loss = 0.0
        # train_predictions = []
        metrics = {}
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if isinstance(model, PreTrainedModel):
                loss = outputs.loss
            else:
                raise Exception('Unknown model type.')
            metrics = {'train/train_loss': loss.item(),
                       'train/epoch': (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,}
            if step % 10 == 0 and step + 1 < n_steps_per_epoch and config['wandb']:
                wandb.log(metrics)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print(f'Finished training epoch {epoch + 1}. ')
        print('Evaluating on validation set...')
        val_metrics = validate_model(model, tokenizer, val_dataloader, 'val')
        print('Evaluating on random sample of train set...')
        train_metrics = validate_model(model, tokenizer, train_dataloader,
                                       'train', num_batches=4)
        if config['wandb'] and epoch % config['num_epochs_per_save'] == 0:
            assert(wandb.run)
            save_path = os.path.join(wandb.run.dir, f'model-epoch-{epoch + 1}')
            print(f'Saving model/tokenizer checkpoint to "{save_path}" ...')
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            wandb.log({**metrics, **train_metrics, **val_metrics})
        else:
            print(f'Train metrics @ epoch {epoch + 1}: {train_metrics | metrics}')
            print(f'Val metrics @ epoch {epoch + 1}: {val_metrics}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-column', type=str, default='gloss',
                        help='Column of dataframe to use as model input (default: \'adult_reformulation_stem\')')
    # parser.add_argument('-o', '--output-column', type=str, default='gloss',
    #                     help='Column of dataframe to use as model target (default: \'child_utterance_stem\')')
    parser.add_argument('-l', '--learning-rate', type=float, default=3e-5, metavar='N',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('-d', '--weight-decay', type=float, default=0.0, metavar='N',
                        help='learning rate (default: 3e-5)')

    parser.add_argument('-b', '--batch-size', type=int, default=224, metavar='N',
                        help='input batch size for training (default: 224)')
    parser.add_argument('-v', '--val-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for evaluation (default: 1024)')
    parser.add_argument('-n', '--num-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 20)')

    parser.add_argument('--num-epochs-per-save', type=int, default=5, metavar='N',
                        help='number of epochs to save model (default: 5)')

    # parser.add_argument('-s', '--max-src-length', type=int, default=16, metavar='N',
    #                     help='max length of source embedding (default: 64)')
    # parser.add_argument('-t', '--max-tgt-length', type=int, default=16, metavar='N',
    #                     help='max length of target embedding (default: 64)')
    # parser.add_argument('-m', '--num-beams', type=int, default=4, metavar='N',
    #                     help='number of beams in beam search decoding (default: 4)')
    # parser.add_argument('--min-age', type=float, default=0, metavar='N',
    #                     help='Minimum age (in months) of child, inclusive')
    # parser.add_argument('--max-age', type=float, default=float('inf'), metavar='N',
    #                     help='Maximum age (in months) of child, exclusive')
    # parser.add_argument('--model-type', type=str, default='transformer',
    #                     help='Model type (either lstm or transformers)')

    # parser.add_argument('--embed-size', type=int, default=128,
    #                     help='Size of embedding layer in LSTM')
    # parser.add_argument('--hidden-size', type=int, default=64,
    #                     help='Size of hidden layer in LSTM')
    # parser.add_argument('--dropout-rate', type=float, default=0.2,
    #                     help='Dropout rate for the LSTM')

    parser.add_argument('--pretrained-model', type=str, default='facebook/bart-base', metavar='MODEL',
                        help=('Name of pretrained Huggingface model/tokenizer to use. '
                              'If in lstm mode, only uses the tokenizer'))
    parser.add_argument('--train-csv-path', type=str,
                        default=os.path.join(data_dir, 'pretraining_df_nt_train.csv'),
                        help='Path to CSV with training data')
    parser.add_argument('--val-csv-path', type=str,
                        default=os.path.join(data_dir, 'pretraining_df_nt_val.csv'),
                        help='Path to CSV with validation data')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Turns off wandb experiment tracking (on by default)')
    # parser.add_argument('--min-utterance-length', type=int, default=1, metavar='N',
    #                     help='Minimum length child utterances to include in train/validation data')
    # parser.add_argument('--min-adult-utterance-length', type=int, default=1, metavar='N',
    #                     help='Minimum length adult utterances to include in train/validation data')
    # parser.add_argument('--prepend-child-age-months', action='store_true',
    #                     help='Whether to prepend the child\'s age (in months) to the inputs')
    # parser.add_argument('--prepend-child-age-years', action='store_true',
    #                     help='Whether to prepend the child\'s age (in years) to the inputs')
    # parser.add_argument('--prepend-child-age-years', action='store_true',
    #                     help='Whether to prepend the child\'s age (in years) to the inputs')

    parser.add_argument('--block-size', type=int, default=64,
                        help='Size to of input "chunks" for MLM pretraining')
    parser.add_argument('--mlm-probability', type=float, default=0.15,
                        help='Probability of masking input token')

    args = parser.parse_args()

    if args.wandb:
        wandb_dir = '/scratch/users/agrawalk/' if os.getenv('USER') == 'agrawalk' else None
        wandb.init(project='word-learning-project', entity='ketan0', dir=wandb_dir, save_code=True)
        wandb.config.update(args) # adds all of the arguments as config variables
        config = wandb.config
    else:
        config = vars(args) # convert Namespace => dict
    run_experiment(config)

if __name__ == '__main__':
    main()
