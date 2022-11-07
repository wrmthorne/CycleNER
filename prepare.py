import os, json, argparse
from random import random
from datasets import Dataset, DatasetDict
from enum import Enum
from transformers import T5ForConditionalGeneration, T5TokenizerFast

class T5(Enum):
    SMALL = 't5-small'
    BASE  = 't5-base'
    LARGE = 't5-large'

def list_all_files(root_dir, file_ext):
    '''Lists all files in a folder and its subfolders that match file_ext

    Args:
        root_dir : top directory
        file_ext : file extensions, e.g. ('.json','.txt') or '.json'
    Returns:
        A list of files in root_dir
    '''
    return sorted([os.path.join(root, name) for root, _, files in os.walk(root_dir) for name in files if name.endswith(file_ext)])

def load_jsons(dir):
    '''Reads in all json files, recursively, in a specified directory
    
    Args:
        dir : Directory to recursively query
    Returns:
        Array of loaded JSONs as dict objects
    '''
    return [json.loads(open(file, 'r').read()) for file in list_all_files(dir, '.json')]


def validate_dataset(dataset, sep_token, bijective=False):
    '''Validates that a user provided dataset meets the format requirements for the CycleNER model

    Args:
        dataset : The dataset, as a dict object, to be inspected
        sep_token : Token used by tokeniser to separate entities and tags
        bijective : Used to determine whether there needs to be a 1:1 mapping between sents and ent_seqs
    '''
    assert list(dataset.keys()) == ['sents', 'ent_seqs'], "Sentences must be keyed as 'sents' and entity sequences must be keyed as 'ent_seqs'"
    assert type(dataset['sents']) == list and type(dataset['ent_seqs']) == list, "Sentences and Entity Sequences must be supplied as a list of strings"
    assert sum([len(ent_seq.split(f' {sep_token} ')) % 2 for ent_seq in dataset['ent_seqs'] if len(ent_seq) > 0]) == 0, "Every entity must have a tag assigned in the format 'entity <sep> tag <sep> entity 2 <sep> tag 2'"

    if bijective:
        assert len(dataset['sents']) == len(dataset['ent_seqs']), 'Dataset must have a bijective mapping of sents to ent_seqs. This dataset does not'

def __define_datasets(jsons):
    '''Splits the datasets into the correct proportional sizes after combining all json datasets into 
    one single dataset.

    jsons : Array of dict objects from json datasets
    '''
    sents = []
    ent_seqs = []

    for file in jsons:
        sents.extend(file['sents'])
        ent_seqs.extend(file['ent_seqs'])

    return sents, ent_seqs

def __build_dataset(s2e_data, s2e_labels, e2s_data, e2s_labels):
    def tokenize(examples):
        return args.tokenizer(
            examples,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
  
    s2e = Dataset.from_dict({
        **tokenize([args.task_prefix + sent for sent in s2e_data]),
        'labels': tokenize(s2e_labels).input_ids})

    e2s = Dataset.from_dict({
        **tokenize([args.task_prefix + ent_seq for ent_seq in e2s_data]),
        'labels': tokenize(e2s_labels).input_ids})

    return DatasetDict({'S2E': s2e, 'E2S': e2s})

def build_pretrain_or_eval_dataset(jsons, save_location, dataset_name):
    '''Constructs a dataset_dict object of Datasets from the provided JSON datasets.

    Args:
        jsons : Array of dict objects as datasets
        save_location : Directory to save the formatted dataset
        dataset_name : Name of the formatted dataset
    Returns:
        File location of saved datasets
    '''
    print('Processing JSONs into dataset...')
    sents, ent_seqs = __define_datasets(jsons)

    dataset_path = os.path.join(save_location, dataset_name)
    pretrain_set = __build_dataset(sents, ent_seqs, ent_seqs, sents)
    pretrain_set.save_to_disk(dataset_path)
    print(f'Saving datasets to {dataset_path}...')

    return dataset_path

def build_train_dataset(jsons, save_location, dataset_name):
    '''Constructs a dataset_dict object of Datasets from the provided JSON datasets.
    First merges all datasets into arrays of sents and ent_seqs and then randomly extracts dev_split
    number of dev instances.

    Args:
        jsons : Array of dict objects as datasets
        save_location : Directory to save the formatted dataset
        dataset_name : Name of the formatted dataset
    Returns:
        File location of saved datasets
    '''
    print('Processing JSONs into dataset...')
    sents, ent_seqs = __define_datasets(jsons)

    dataset_path = os.path.join(save_location, dataset_name)
    train_dataset = __build_dataset(sents, sents, ent_seqs, ent_seqs)
    train_dataset.save_to_disk(dataset_path)
    print(f'Saving datasets to {dataset_path}...')

    return dataset_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CycleNER model for training.')
    parser.add_argument('--model', default='small', help='T5 Model type (SMALL, BASE, LARGE).')
    parser.add_argument('--model_name', default=None, help='Name of this specific version of the model. If nothing is provided, the model type will be used (e.g. SMALL).')
    parser.add_argument('--model_dir', default='./models', help='Location to store the model and all associated files.')
    parser.add_argument('--model_max_length', default=128, help='Max length for model generations and tokenized strings.')
    parser.add_argument('--sep_token', default='|', help='Separator token to use in T5 model.')
    parser.add_argument('--task_prefix', default='Detect tags', help='Task prefix for T5 model to learn.')
    parser.add_argument('--train_dir', default='./data/train', help='Directory containing training json data to load into dataset.')
    parser.add_argument('--eval_dir', default=None, help='Directory containing evaluation json data to load into dataset. If not supplied, evaluation cannot be conducted in training script.')
    parser.add_argument('--test_dir', default='./data/test', help='Directory containing testing json data to load into dataset.')
    args = parser.parse_args()

    # Define the T5 model to use
    args.model = T5[args.model.upper()]

    # Define the model directory
    if args.model_name is not None:
        args.model_dir = os.path.join(args.model_dir, args.model_name)
    else:
        args.model_dir = os.path.join(args.model_dir, args.model.name)
        args.model_name = args.model.name

    # Download the relevant model and tokenizer
    s2e = T5ForConditionalGeneration.from_pretrained(args.model.value)
    e2s = T5ForConditionalGeneration.from_pretrained(args.model.value)
    args.tokenizer = T5TokenizerFast.from_pretrained(args.model.value, model_max_length=args.model_max_length)
    args.tokenizer.sep_token = args.sep_token

    # Append ': ' to the prefix
    if not args.task_prefix.strip().endswith(':'):
        args.task_prefix = args.task_prefix + ': '

    datasets_dir = os.path.join(args.model_dir, 'data')

    # Validate and load all training datasets
    assert os.path.exists(args.train_dir)
    for file in (json_files := load_jsons(args.train_dir)):
        validate_dataset(file, args.sep_token)
    train_location = build_train_dataset(json_files, datasets_dir, 'train_dataset')

    eval_dataset = None
    if args.eval_dir is not None:
        assert(os.path.exists(args.eval_dir))
        for file in (json_files := load_jsons(args.eval_dir)):
            validate_dataset(file, args.sep_token, bijective=True)
        eval_location = build_pretrain_or_eval_dataset(json_files, datasets_dir, 'eval_dataset')

    assert os.path.exists(args.test_dir)
    for file in (json_files := load_jsons(args.test_dir)):
        validate_dataset(file, args.sep_token, bijective=True)
    test_location = build_pretrain_or_eval_dataset(json_files, datasets_dir, 'test_dataset')

    # Save models and tokenizer to model_dir
    s2e.save_pretrained(os.path.join(args.model_dir, 'S2E'))
    e2s.save_pretrained(os.path.join(args.model_dir, 'E2S'))
    args.tokenizer.save_pretrained(os.path.join(args.model_dir, 'tokenizer'))
            
    config = {
        'model': args.model.value,
        'model_name': args.model_name,
        'model_dir': args.model_dir,
        'train_dataset': train_location,
        'eval_dataset': eval_location,
        'test_dataset': test_location,
        'task_prefix': args.task_prefix,
        'sep_token': args.sep_token,
    }

    with open(os.path.join(args.model_dir, 'config.json'), 'w+') as file:
        json.dump(config, file, indent=4)

    print(f'Preparation complete. All files can be found in {args.model_dir}')
    

