import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, DistilBertConfig
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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def __getitem__(self, idx):
        x = self.tokenizer(self.x[idx], truncation=True, padding='max_length', return_tensors="pt")
        for k, v in x.items():
            x[k] = v.squeeze(0)
        return x

    def __len__(self):
        return len(self.x)

def decode_text(decoder, logits):
    score = F.softmax(logits, dim=-1)
    idx = torch.argmax(score, dim=-1)
    idx = idx[:, 1:]
    idx = idx.tolist()
    res = []
    for seq in idx:
        res.append(decoder.decode(seq[:seq.index(decoder.sep_token_id)]))
    return res

def test(informal):
    if torch.cuda.is_available():
        device = torch.device('cuda:3')
        print(f'Using GPU device: {device}')
    else:
        device = torch.device('cpu')
        print(f'GPU is not available, using CPU device {device}')
    
    test_config = {'batch_size':5, 'epoch':29, 'save_dir':'./checkpoints/'}
    
    test_dataset = FormalDataset(informal)
    dataloader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, drop_last=False)
    config = DistilBertConfig()
    model = DistilBertForMaskedLM(config)
    load_model(test_config['epoch'], model, test_config['save_dir'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            inp = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            logits = model(input_ids=inp, attention_mask=attn)[0]
            preds = decode_text(test_dataset.tokenizer, logits)
            for seq in preds:
                with open('test_pred.txt', 'a') as res_file:
                    res_file.writelines(seq+'\n')
                


if __name__=='__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    path = './GYAFC_Corpus/Entertainment_Music'
    informal = load_dataset(path)
    
    test(informal)
