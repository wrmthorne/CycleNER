import torch
import os
import json
from transformers import T5ForConditionalGeneration, T5TokenizerFast

class CycleNER:
    def __init__(self, s2e, e2s, tokenizer, device='cpu', **kwargs):
        self.prefix = kwargs.pop('task_prefix', None)

        self.device = device

        self.s2e = s2e.to(self.device)
        self.e2s = e2s.to(self.device)

        self.tokenizer = tokenizer
        self.prefix_ids = tokenizer(self.prefix, return_tensors='pt').input_ids.to(self.device)
        self.cycle = 'S'

    def __call__(self, input_ids, attention_mask, labels):
        '''
        Corresponds to the forward pass of the appropriate model, set using model.s_cycle() or
        model.e_cycle()
        '''
        if self.cycle == 'S':
            return self._cycle(input_ids, attention_mask, labels, self.s2e, self.e2s)
        else:
            return self._cycle(input_ids, attention_mask, labels, self.e2s, self.s2e)

    def _cycle(self, input_ids, attention_mask, labels, generating_model, training_model):
        '''
        Generic cycle function which performs each stage for S or E cycle. The behaviour of each
        cycle is, hence, defined:
            S-Cycle:
                1) From seed sentence s, generate entity sequence q'
                2) From q', generate something that resembles s, named s'
                3) Calculate reconstruction loss between original sentence s and generated sentence s'
                4) Back propagate loss
            
            E-Cycle:
                1) From seed entity sequence q, generate a sentence s'
                2) From s', generate something that resembles q, named q'
                3) Calculate reconstruction loss between original sequence q and generated sequence q'
                4) Back propagate loss
        '''
        input_prime_ids = generating_model.generate(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            max_length     = self.tokenizer.model_max_length
        )

        # Skip decode-encode by removing leading pad and generating an attention mask
        input_prime_ids = input_prime_ids[:, 1:]

        # Insert prefix ids into sequence for training model
        input_prime_ids = torch.cat((self.prefix_ids.repeat(input_prime_ids.size(0), 1), input_prime_ids.to(self.device)), dim=1)

        # Create attention mask
        attention_mask = input_prime_ids.clone()
        attention_mask[attention_mask != 0] = 1

        # Set pad tokens to be ignored by PyTorch loss function
        labels[labels==self.tokenizer.pad_token_id] = -100
        
        # Generate tokens and calculate reconstruction loss
        outputs = training_model(
            input_ids      = input_prime_ids.to(self.device),
            attention_mask = attention_mask.to(self.device),
            labels         = labels.to(self.device)
        )

        return outputs

    def generate(self, input_ids):
        '''
        Uses S2E model generate function to tag a given input sequence.
        '''
        return self.s2e.generate(
            input_ids      = input_ids,
            max_length     = self.tokenizer.model_max_length
        )

    def train(self):
        '''Tells PyTorch that each model is about to be trained'''
        self.s2e.train()
        self.e2s.train()

    def eval(self):
        '''Tells PyTorch that each model is about to be evaluated'''
        self.s2e.eval()
        self.e2s.eval()

    def s_cycle(self):
        '''Sets the learning cycle to S'''
        self.cycle = 'S'

    def e_cycle(self):
        '''Sets the learning cycle to E'''
        self.cycle = 'E'

    def zero_grad(self):
        '''Zeroes the gradients of all parameters in both sub-models'''
        self.s2e.zero_grad()
        self.e2s.zero_grad()

    def to(self, device):
        '''Moves all of the sub-model parameters and tensor variables to the specified device'''
        self.device = device
        self.s2e.to(self.device)
        self.e2s.to(self.device)
        self.prefix_ids = self.prefix_ids.to(self.device)

    def save_pretrained(self, save_directory):
        '''Saves the model state and internal variables to the specified directory'''
        self.s2e.save_pretrained(os.path.join(save_directory, 'S2E'))
        self.e2s.save_pretrained(os.path.join(save_directory, 'E2S'))
        self.tokenizer.save_pretrained(os.path.join(save_directory, 'tokenizer'))

        config = {'task_prefix': self.prefix}
        with open(os.path.join(save_directory, 'config.json'), 'w+') as file:
            config = json.dump(config, file, indent=4)

    @classmethod
    def from_pretrained(cls, model_path):
        '''Loads a pretrained model state from the specified directory'''
        s2e = T5ForConditionalGeneration.from_pretrained(os.path.join(model_path, 'S2E'))
        e2s = T5ForConditionalGeneration.from_pretrained(os.path.join(model_path, 'E2S'))
        tokenizer = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer'))

        with open(os.path.join(model_path, 'config.json'), 'r') as file:
            config = json.load(file)

        return cls(s2e, e2s, tokenizer, **config)
