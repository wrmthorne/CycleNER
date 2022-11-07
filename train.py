'''
2) Load checkpoint
6) Early stopping criterion
7) LR scheduler
8) Save args, optimiser state and lr-chedule with checkpoint
'''

import argparse
import os
import json
import logging
import random
import numpy as np
import torch
import gc
from datasets import load_from_disk
from tqdm import tqdm, trange
from CycleNER import CycleNER
from torch.optim import AdamW
import torch.nn.functional as F
from itertools import chain, zip_longest
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
import datasets

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
datasets.logging.set_verbosity_error()

# Code from https://stackoverflow.com/a/73704579/8145428
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, checkpoint_name):
    output_dir = os.path.join(args.output_dir, 'checkpoints', f'checkpoint-{checkpoint_name}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)


def calculate_f1(decoded_outputs, labels, tokenizer):
    split_term = f' {tokenizer.sep_token} '

    # Process labels into array of tagged terms
    y_true = list(map(lambda x: [split_term.join(x_i) for x_i in zip(x.split(split_term)[0::2], x.split(split_term)[1::2])], labels))
    y_pred = list(map(lambda x: [split_term.join(x_i) for x_i in zip(x.split(split_term)[0::2], x.split(split_term)[1::2])], decoded_outputs))

    binarizer = MultiLabelBinarizer()
    binarizer.fit(y_true + y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        binarizer.transform(y_true), 
        binarizer.transform(y_pred), 
        average='micro'
        )

    return precision, recall, f1


def evaluate(eval_dataset, model, tokenizer, batch_size, model_name=None, calc_f1=False):
    logger.info(f"***** Running {model_name} evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch Size = %d", batch_size)

    ev_loss = 0

    batch_indices = [list(range(len(eval_dataset)))[i:i+batch_size] for i in range(0, len(eval_dataset), batch_size)]

    model.eval()

    generated_outputs = []

    for batch_idxs in batch_indices:
        batch = eval_dataset.select(batch_idxs
        )
        
        input_ids, attention_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']

        model.zero_grad()

        # Ensure that pad tokens are ignored in loss calculation
        label_ids[label_ids==tokenizer.pad_token_id] = -100

        loss = model(
            input_ids      = input_ids.to(args.device),
            attention_mask = attention_mask.to(args.device),
            labels         = label_ids.to(args.device)
        ).loss

        if calc_f1:
            generated_output = model.generate(
                input_ids  = input_ids.to(args.device),
                max_length = tokenizer.model_max_length
                )

            generated_outputs.extend(F.pad(generated_output, (tokenizer.model_max_length - generated_output.size(1), 0)))

        ev_loss += loss.item()

    f1_output = {}
    if calc_f1:
        decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        decoded_labels  = tokenizer.batch_decode(eval_dataset['labels'], skip_special_tokens=True)
        precision, recall, f1 = calculate_f1(decoded_outputs, decoded_labels, tokenizer)
        f1_output = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return ev_loss / len(batch_idxs), f1_output

def train(s2e_dataset, e2s_dataset, model, tokenizer, s2e_optimiser, e2s_optimiser):
    logger.info("***** Running training *****")
    logger.info("  Num S-cycle examples = %d", len(s2e_dataset))
    logger.info("  Num E-cycle examples = %d", len(e2s_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch Size = %d", args.batch_size)

    epochs_trained = 0
    global_step = 0

    if args.early_stopping:
        stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    cycle_info = {
        'S': {'name': 'S2E', 'loss': 0, 'step': 0},
        'E': {'name': 'E2S', 'loss': 0, 'step': 0}
        }

    s2e_indices = [list(range(len(s2e_dataset)))[i:i+args.batch_size] for i in range(0, len(s2e_dataset), args.batch_size)]
    e2s_indices = [list(range(len(e2s_dataset)))[i:i+args.batch_size] for i in range(0, len(e2s_dataset), args.batch_size)]

    # TODO: MAYBE COME BACK AND REDO THIS - FOR UNEAVEN S AND E CYCLE DATA, CURRENTLY ITERATES BETWEEN S AND E CYCLE AND THEN ONLY RUNS
    # THE CYCLE WITH MORE DATA UNTIL THE WHOLE EPOCH IS COMPLETE. MAYBE TRY TO EVENLY INTERSPERSE THE CYCLES?
    s_cycles = ['S'] * len(s2e_indices)
    e_cycles = ['E'] * len(e2s_indices)
    cycles = [x for x in chain.from_iterable(zip_longest(s_cycles, e_cycles)) if x is not None]

    train_iterator = trange(epochs_trained, int(args.epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(cycles, desc="Iteration")
        s_cycle_iterator = iter(tqdm(s2e_indices, desc='S Batches'))
        e_cycle_iterator = iter(tqdm(e2s_indices, desc='E Batches'))

        # Shuffle datasets
        s2e_dataset.shuffle(seed=42)
        e2s_dataset.shuffle(seed=0)

        model.train()

        for cycle in epoch_iterator:
            if cycle == 'S':
                model.s_cycle()
                optimiser = e2s_optimiser
                batch = s2e_dataset.select(next(s_cycle_iterator))
            else:
                model.e_cycle()
                optimiser = s2e_optimiser
                batch = e2s_dataset.select(next(e_cycle_iterator))

            input_ids, attention_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']
            model.zero_grad()

            loss = model(
                input_ids      = input_ids.to(args.device),
                attention_mask = attention_mask.to(args.device),
                labels         = label_ids.to(args.device)
            ).loss

            loss.backward()
            optimiser.step()
            cycle_info[cycle]['loss'] += loss.item()

            global_step += 1
            cycle_info[cycle]['step'] += 1

            if global_step % args.save_steps == 0 and args.save_steps != -1 and not args.no_save:
                save_checkpoint(model, global_step)

            if global_step % args.logging_steps == 0:
                for info in cycle_info.values():
                    # The total loss of each model is divided by the step of the respective model but recorded at a global step
                    logger.info(f'{info["name"]} loss: {info["loss"] / info["step"]}')
                    tb_writer.add_scalar(f'train_{info["name"]}', info["loss"] / info["step"], global_step)

        if not args.no_eval:
            s2e_ev_loss, f1_output = evaluate(eval_dataset['S2E'], model.s2e, tokenizer, args.batch_size, model_name='S2E', calc_f1=True)
            tb_writer.add_scalar("eval_S2E", s2e_ev_loss, global_step)
            tb_writer.add_scalar('f1_S2E', f1_output['f1'], global_step)
            e2s_ev_loss, _ = evaluate(eval_dataset['E2S'], model.e2s, tokenizer, args.batch_size, model_name='E2S')
            tb_writer.add_scalar("eval_E2S", e2s_ev_loss, global_step)

        if not args.no_save:
            save_checkpoint(model, global_step)

        if args.early_stopping and stopper.early_stop(s2e_ev_loss):
            logger.info(f'Stopping training as early stopping criteria is met...')
            break

    return cycle_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleNER training script.')
    parser.add_argument('--seed', default=42, help='Seed to use for randomness in the model.', type=int)
    parser.add_argument('--no-cuda', default=False, help='Disable use of CUDA i.e. force training on CPU.', action='store_true')
    parser.add_argument('--no-eval', default=False, help='Disable evaluation using eval set when training. Will be automatically set to true if no eval set exists.', action='store_true')
    parser.add_argument('--no-train', default=False, help='Disable training of models', action='store_true')
    parser.add_argument('--early_stopping', default=False, help='Enable early stopping for if S2E validation loss consistently increases.', action='store_true')
    parser.add_argument('--patience', default=3, help='Number of consecutive epochs that the validation loss can increase before early stopping is applied.')
    parser.add_argument('--min_delta', default=0, help='Minimum amount of increase in validation loss over previous epoch to count against patience epochs.')
    parser.add_argument('--logging_steps', default=100, help='Number of training steps to take before logging output.', type=int)
    parser.add_argument('--save_steps', default=100, help='Number of training iterations to perform before saving a checkpoint. Using -1 will only save checkpoints at the end of each epoch.', type=int)
    parser.add_argument('--no-save', default=False, help='Set flag to prevent checkpoints being saved', action='store_true')
    parser.add_argument('--model_dir', default='./models/SMALL', help='Directory to find the CycleNER model files to be used in training.')
    parser.add_argument('--output_dir', default=None, help='Directory to store checkpoints made during runtime.')
    parser.add_argument('--epochs', default=10, help='Number of epochs to train S2E and E2S over.', type=int)
    parser.add_argument('--s2e_lr', default=1e-4, help='Learning rate for S2E model.', type=float)
    parser.add_argument('--e2s_lr', default=1e-4, help='Learning rate for E2S model.', type=float)
    parser.add_argument('--batch_size', default=32, help='Batch size for training S2E and E2S models.', type=int)
    parser.add_argument('--summary_writer_dir', default=None, help='Directory to store the output of summary writer.')
    args = parser.parse_args()

    set_seed(args.seed)
    args.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    assert os.path.exists(args.model_dir), f'No model directory found at {args.model_dir}'

    with open(os.path.join(args.model_dir, 'config.json'), 'r') as file:
        config = json.load(file)

    if args.output_dir is None:
        args.output_dir = args.model_dir

    # Load pretrained CycleNER model at specified path
    cycle_ner = CycleNER.from_pretrained(args.model_dir)
    cycle_ner.to(args.device)

    # Set up logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.info(f'Training Args {args}')
    tb_writer = SummaryWriter(log_dir=args.summary_writer_dir)

    if config['eval_dataset'] == None:
        args.no_eval = True

    eval_dataset = load_from_disk(config['eval_dataset']) if not args.no_eval else None
    if eval_dataset is not None:
        eval_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])

    if not args.no_train:
        s2e_optimiser = AdamW(cycle_ner.s2e.parameters(), lr=args.s2e_lr)
        e2s_optimiser = AdamW(cycle_ner.e2s.parameters(), lr=args.e2s_lr)
        train_dataset = load_from_disk(config['train_dataset'])
        train_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
        train(train_dataset['S2E'], train_dataset['E2S'], cycle_ner, cycle_ner.tokenizer, s2e_optimiser, e2s_optimiser)

    torch.cuda.empty_cache()
    gc.collect()

