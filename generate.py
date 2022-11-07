import argparse
from CycleNER import CycleNER
from pprint import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleNER generation script.')
    parser.add_argument('--model_dir', default='./models', help='Location to store the model and all associated files.')
    parser.add_argument('--input', default=None, help='Sequence to tag using S2E model.')
    args = parser.parse_args()

    assert args.input is not None

    cycle_ner = CycleNER.from_pretrained(args.model_dir)
    tokenizer = cycle_ner.tokenizer
    task_prefix = cycle_ner.prefix
    input_string = task_prefix + args.input

    input_ids = tokenizer(
        input_string,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
        ).input_ids

    encoded_output = cycle_ner.s2e.generate(input_ids=input_ids, max_length=tokenizer.model_max_length)

    decoded_output = tokenizer.batch_decode(encoded_output, skip_special_tokens=True)
    pprint(decoded_output)