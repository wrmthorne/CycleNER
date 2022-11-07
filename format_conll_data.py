from datasets import load_dataset
import argparse
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
import json
import random


# Dictionary mapping BIO tags to compound tags and then to word based tags
tag_to_idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
tag_to_string = {'PER': 'person', 'ORG': 'organisation', 'LOC': 'location', 'MISC': 'miscellaneous'}
string_to_tag = {string: tag for tag, string in tag_to_string.items()}

def save_split(data, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, 'dataset.json')

    with open(filename, 'w+') as file:
        json.dump(data, file, indent=4)

    return filename

def build_ent_sequences(tokens, tags, sep_token):
    '''Combines BIO tags used in the CoNLL2003 dataset into combined PER, ORG, LOC, MISC labels.
    Also splits sequence into expected entity sequence format.

    Args:
        tokens : Array of tokens
        tags : Array of tags in BIO format
        sep_token : Token used by tokeniser to separate entities and tags
    Returns:
        Array of entity sequences
    '''
    compound_tokens = []

    for token, tag in zip(tokens, tags):
        if tag in [1, 3, 5, 7]:
            compound_tokens.append([token, sep_token, f"{tag_to_string[idx_to_tag[tag].split('-')[-1]]}"])
        elif tag in [2, 4, 6, 8]:
            compound_tokens[-1][-2:-2] = [token]
    
    return [' '.join(token_tags) for token_tags in compound_tokens]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleNER training script.')
    parser.add_argument('--train_dir', default='./data/train', help='Where to save the train dataset.')
    parser.add_argument('--eval_dir', default='./data/eval', help='Where to save the eval dataset.')
    parser.add_argument('--test_dir', default='./data/test', help='Where to save the test dataset.')
    parser.add_argument('--s_examples', default=-1, help='Number of training sentences to use. -1 is all sentences.', type=int)
    parser.add_argument('--e_examples', default=-1, help='Number of training entity sequences to use. -1 is all entity sequences.', type=int)
    parser.add_argument('--sep_token', default='|', help='Separator token used in T5 Models.')
    args = parser.parse_args()

    dataset = load_dataset("conll2003")
    
    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda batch: (
                sents := TreebankWordDetokenizer().detokenize(batch['tokens']),
                ent_seqs := f' {args.sep_token} '.join(build_ent_sequences(batch['tokens'], batch['ner_tags'], args.sep_token)),
                {
                    'sents': sents,
                    'ent_seqs': ent_seqs,
                })[-1]).remove_columns(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])


    dataset = dataset.shuffle(seed=42)

    args.s_examples = len(dataset['train']) if args.s_examples == -1 else args.s_examples
    args.e_examples = len(dataset['train']) if args.e_examples == -1 else args.e_examples

    random.seed(42)
    train_examples = {
        'sents': random.sample(dataset['train']['sents'], args.s_examples),
        'ent_seqs': random.sample(dataset['train']['ent_seqs'], args.e_examples)
    }

    train_file = save_split(train_examples, args.train_dir)
    eval_file = save_split(dataset['validation'].to_dict(), args.eval_dir)
    test_file  = save_split(dataset['test'].to_dict(), args.test_dir)

    print(f'Datasets saved as: \nTRAIN: {train_file} \nEVAL: {eval_file} \nTEST: {test_file}')
