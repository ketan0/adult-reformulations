#!/usr/bin/env python3

import wandb
import argparse
import os
from typing import Union
import torch
from torch import nn
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.optimization import get_scheduler
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto.configuration_auto import AutoConfig
from tqdm import tqdm
from datasets.load import load_metric
import nltk

from nmt_model import NMT

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

class SequenceToSequenceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def postprocess_text(text: list[str]):
    text = [pred.strip() for pred in text]
    # rougeLSum expects newline after each sentence
    text = ['\n'.join(nltk.sent_tokenize(pred)) for pred in text]
    return text

def log_preds_gt_table(decoded_inputs, decoded_preds, decoded_labels, table_name):
    """🐝 Log a wandb.Table with (inpt, pred, target)"""
    table = wandb.Table(columns=['input (parent reformulation)', 'prediction', 'target (child utterance)'])
    for inpt, pred, targ in zip(decoded_inputs, decoded_preds, decoded_labels):
        table.add_data(inpt, pred, targ)
    wandb.log({table_name: table}, commit=False)

def validate_model(model: nn.Module, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                   dataloader: DataLoader, metric_prefix: str, config: dict, pred_table_name: str, num_batches=None):
    model.eval()
    gen_kwargs = {
        'max_length': config['max_tgt_length'],
        'num_beams': config['num_beams'],
    }
    metric = load_metric('rouge')
    all_inputs = []
    all_preds = []
    all_labels = []
    for i, batch in enumerate(tqdm(dataloader)):
        if num_batches is not None and i == num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if isinstance(model, PreTrainedModel):
                generated_tokens = model.generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs
                )
            elif isinstance(model, NMT):
                generated_hyps = [model.beam_search(input_ids, attention_mask,
                                                    beam_size=config['num_beams'])
                                    for input_ids, attention_mask
                                    in zip(batch['input_ids'], batch['attention_mask'])]
                generated_tokens = [torch.tensor(hyps[0].value) for hyps in generated_hyps]
                assert(tokenizer.pad_token_id)
                generated_tokens = nn.utils.rnn.pad_sequence(generated_tokens, batch_first=True,
                                                             padding_value=tokenizer.pad_token_id)
                generated_tokens = generated_tokens.int()
            else:
                raise Exception('Model has unexpected type')
        labels = batch['labels']
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # print('input ids:', batch['input_ids'])
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        # print('Generated tokens:', generated_tokens)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # print('Decoded preds:', decoded_preds)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = postprocess_text(decoded_preds)
        decoded_labels = postprocess_text(decoded_labels)
        decoded_inputs = postprocess_text(decoded_inputs)

        all_inputs.extend(decoded_inputs)
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)
        if i == 0 and config['wandb']:
            log_preds_gt_table(decoded_inputs, decoded_preds, decoded_labels, pred_table_name)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    val_df = pd.DataFrame({'input': all_inputs, 'pred': all_preds, 'label': all_labels})
    val_df.to_csv(os.path.join(data_dir, config['output_val_csv']), index=False)

    metrics = metric.compute(use_stemmer=True)
    assert(metrics is not None)
    # Extract a few results from ROUGE
    metrics = {key: value.mid.fmeasure * 100 for key, value in metrics.items()}
    metrics = {metric_prefix + k: round(v, 4) for k, v in metrics.items()}
    return metrics

def run_experiment(config: dict):
    df_train = pd.read_csv(config['train_csv_path'])
    df_val = pd.read_csv(config['val_csv_path'])
    # make type checker happy
    assert(isinstance(df_train, pd.DataFrame) and isinstance(df_val, pd.DataFrame))
    age_query = f'target_child_age >= {config["min_age"]} and target_child_age < {config["max_age"]}'
    utt_len_query = f'gloss_len >= {config["min_utterance_length"]}'
    filter_query = f'{age_query} and {utt_len_query}'
    df_train = df_train.query(filter_query)
    df_val = df_val.query(filter_query)
    assert(isinstance(df_train, pd.DataFrame) and isinstance(df_val, pd.DataFrame))
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    print(f'Size of training set: {len(df_train)}')
    print(f'Size of validation set: {len(df_val)}')

    train_source_texts = list(df_train[config['input_column']])
    train_target_texts = list(df_train[config['output_column']])
    val_source_texts = list(df_val[config['input_column']])
    val_target_texts = list(df_val[config['output_column']])
    print('Tokenizing inputs...')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base',
        config=AutoConfig.from_pretrained(config['pretrained_model'], max_position_embeddings=config['max_src_length']))
    if (not isinstance(tokenizer, PreTrainedTokenizerFast) and
        not isinstance(tokenizer, PreTrainedTokenizer)):
        raise Exception('not a proper tokenizer')
    train_source_encodings = tokenizer(train_source_texts, truncation=True,
                                       max_length=config['max_src_length'], padding='max_length')
    val_source_encodings = tokenizer(val_source_texts, truncation=True,
                                     max_length=config['max_src_length'], padding='max_length')
    with tokenizer.as_target_tokenizer():
        train_target_encodings = tokenizer(train_target_texts, truncation=True, max_length=config['max_tgt_length'], padding='max_length')
        train_target_labels = train_target_encodings['input_ids']

        val_target_encodings = tokenizer(val_target_texts, truncation=True, max_length=config['max_tgt_length'], padding='max_length')
        val_target_labels = val_target_encodings['input_ids']
    train_dataset = SequenceToSequenceDataset(train_source_encodings, train_target_labels)
    val_dataset = SequenceToSequenceDataset(val_source_encodings, val_target_labels)
    assert(tokenizer.pad_token_id is not None and tokenizer.eos_token_id is not None)
    if config['model_type'] == 'transformer':
        model = AutoModelForSeq2SeqLM.from_pretrained(config['pretrained_model'])
    elif config['model_type'] == 'lstm':
        model = NMT(config['embed_size'], config['hidden_size'], tokenizer.vocab_size,
                    tokenizer.pad_token_id,
                    tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    dropout_rate=config['dropout_rate'])
    else:
        raise Exception('Unexpected value for --model-type provided')
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    num_workers = 0
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'],
                                  collate_fn=data_collator, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config['val_batch_size'],
                                collate_fn=data_collator, num_workers=num_workers)

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
    # progress_bar = tqdm(range(num_training_steps))
    print('Starting training.')
    print('Evaluating on validation set...')
    val_metrics = validate_model(model, tokenizer, val_dataloader, 'val/', config, 'val_predictions_table')
    if config['wandb']:
        wandb.log({'train/epoch': 0, **val_metrics})
    else:
        print(f'Val metrics @ epoch 0: {val_metrics}')
    # n_steps_per_epoch = len(train_dataloader)
    # for epoch in range(config['num_epochs']):
    #     print(f'Training epoch {epoch + 1}...')
    #     model.train()
    #     train_loss = 0.0
    #     # train_predictions = []
    #     metrics = {}
    #     for step, batch in enumerate(tqdm(train_dataloader)):
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         if isinstance(model, PreTrainedModel):
    #             loss = outputs.loss
    #         elif isinstance(model, NMT):
    #             loss = outputs
    #         else:
    #             raise Exception('Unknown model type.')
    #         metrics = {'train/train_loss': loss.item(),
    #                    'train/epoch': (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,}
    #         if step % 10 == 0 and step + 1 < n_steps_per_epoch and config['wandb']:
    #             wandb.log(metrics)
    #         train_loss += loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #     print(f'Finished training epoch {epoch + 1}. ')
    #     print('Evaluating on validation set...')
    #     val_metrics = validate_model(model, tokenizer, val_dataloader,
    #                                  'val/', config,  'val_predictions_table')
    #     print('Evaluating on random sample of train set...')
    #     train_metrics = validate_model(model, tokenizer, train_dataloader,
    #                                    'train/', config, 'train_predictions_table', num_batches=4)
    #     if config['wandb']:
    #         assert(wandb.run)
    #         print('Saving model checkpoint...')
    #         save_path = os.path.join(wandb.run.dir, f'model-epoch-{epoch + 1}')
    #         if isinstance(model, PreTrainedModel):
    #             model.save_pretrained(save_path)
    #         else:
    #             torch.save(model.state_dict(), save_path + '.pt')
    #         wandb.log({**metrics, **train_metrics, **val_metrics})
    #     else:
    #         print(f'Train metrics @ epoch {epoch + 1}: {train_metrics | metrics}')
    #         print(f'Val metrics @ epoch {epoch + 1}: {val_metrics}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-column', type=str, default='parent_gloss',
                        help='Column of dataframe to use as model input (default: \'adult_reformulation_stem\')')
    parser.add_argument('-o', '--output-column', type=str, default='gloss',
                        help='Column of dataframe to use as model target (default: \'child_utterance_stem\')')
    parser.add_argument('-l', '--learning-rate', type=float, default=3e-5, metavar='N',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('-b', '--batch-size', type=int, default=224, metavar='N',
                        help='input batch size for training (default: 224)')
    parser.add_argument('-v', '--val-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for evaluation (default: 1024)')
    parser.add_argument('-n', '--num-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('-s', '--max-src-length', type=int, default=16, metavar='N',
                        help='max length of source embedding (default: 64)')
    parser.add_argument('-t', '--max-tgt-length', type=int, default=16, metavar='N',
                        help='max length of target embedding (default: 64)')
    parser.add_argument('-m', '--num-beams', type=int, default=4, metavar='N',
                        help='number of beams in beam search decoding (default: 4)')
    parser.add_argument('--min-age', type=float, default=0, metavar='N',
                        help='Minimum age (in months) of child, inclusive')
    parser.add_argument('--max-age', type=float, default=float('inf'), metavar='N',
                        help='Maximum age (in months) of child, exclusive')
    parser.add_argument('--model-type', type=str, default='transformer',
                        help='Model type (either lstm or transformers)')

    parser.add_argument('--embed-size', type=int, default=128,
                        help='Size of embedding layer in LSTM')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Size of hidden layer in LSTM')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for the LSTM')

    parser.add_argument('--pretrained-model', type=str, default='facebook/bart-base', metavar='MODEL',
                        help=('Name of pretrained Huggingface model/tokenizer to use. '
                              'If in lstm mode, only uses the tokenizer'))
    parser.add_argument('--train-csv-path', type=str,
                        default=os.path.join(data_dir, 'Apr112022_child_adult_reformulations_nonull_train.csv'),
                        help='Path to CSV with training data')
    parser.add_argument('--val-csv-path', type=str,
                        default=os.path.join(data_dir, 'Apr112022_child_adult_reformulations_nonull_val.csv'),
                        help='Path to CSV with validation data')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Turns off wandb experiment tracking (on by default)')
    parser.add_argument('--min-utterance-length', type=int, default=1, metavar='N',
                        help='Minimum length utterances to include in train/validation data')
    parser.add_argument('--output-val-csv', type=str, default='2May2022-whole-val-preds.csv',
                        help='Where to output full list of val preds')
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
