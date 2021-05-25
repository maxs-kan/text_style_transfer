import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

import codecs
import random
import numpy as np
import os
from tqdm import tqdm
import copy
import wandb
import pandas as pd

def load_dataset(data_dir, phase='test'):
    xpath = os.path.join(data_dir, phase, 'informal')
    with codecs.open(xpath, 'r', encoding='utf-8') as inp:
        informal = [s.strip('\n') for s in inp.readlines()]

    return informal

def load_model(epoch, model, model_dir_path):
    save_path = os.path.join(model_dir_path, '{}.pt'.format(epoch))
    checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

class FormalDataset(Dataset):
    def __init__(self, informal):
        super(FormalDataset, self).__init__()
        self.x = informal
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    def __getitem__(self, idx):
        x = self.tokenizer(self.x[idx], truncation=True, padding='max_length', return_tensors="pt")
        for k, v in x.items():
            x[k] = v.squeeze(0)
        return x

    def __len__(self):
        return len(self.x)

def decode_text(model, decoder, inp):
    idx = model.generate(inp, num_beams=4, max_length=50, early_stopping=True, eos_token_id=decoder.eos_token_id)
    idx = [decoder.decode(g, skip_special_tokens=True) for g in idx] #, clean_up_tokenization_spaces=False
    return idx

def test(informal):
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print(f'Using GPU device: {device}')
    else:
        device = torch.device('cpu')
        print(f'GPU is not available, using CPU device {device}')
    
    test_config = {'batch_size':5, 'epoch':5, 'save_dir':'./checkpoints/'}
    
    test_dataset = FormalDataset(informal)
    dataloader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, drop_last=False)
#     config = BartConfig()
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    load_model(test_config['epoch'], model, test_config['save_dir'])
    model.config.decoder_start_token_id = test_dataset.tokenizer.cls_token_id
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            inp = batch['input_ids'].to(device)
            preds = decode_text(model, dataloader.dataset.tokenizer, inp)
            for seq in preds:
                with open('test_pred.txt', 'a') as res_file:
                    res_file.writelines(seq+'\n')
                


if __name__=='__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    path = './GYAFC_Corpus/Family_Relationships/'
    informal = load_dataset(path)
    
    test(informal)
