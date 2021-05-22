import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
from torch.utils.data import Dataset, DataLoader

import codecs
import random
import numpy as np
import os
from tqdm import tqdm
import copy
import wandb
import pandas as pd

def load_dataset(data_dir, phase='train'):
    xpath = os.path.join(data_dir, phase, 'informal')
    with codecs.open(xpath, 'r', encoding='utf-8') as inp:
        informal = [s.strip('\n') for s in inp.readlines()]

    ypath = os.path.join(data_dir, phase, 'formal')
    with codecs.open(ypath, 'r', encoding='utf-8') as inp:
        formal = [s.strip('\n') for s in inp.readlines()]
    
    assert len(formal) == len(informal)

    return formal, informal

def save_checkpoint(epoch, model, lr_scheduler, optimizer, model_dir_path):
    save_path = os.path.join(model_dir_path, '{}.pt'.format(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
    }, save_path)
    print(f'Saved checkpoint to {save_path}')

def load_model(epoch, model, lr_scheduler, optimizer, model_dir_path):
    save_path = os.path.join(model_dir_path, '{}.pt'.format(epoch))
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class FormalDataset(Dataset):
    def __init__(self, informal, formal):
        super(FormalDataset, self).__init__()
        self.x = informal
        self.y = formal
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def __getitem__(self, idx):
        x = self.tokenizer(self.x[idx], truncation=True, padding='max_length', return_tensors="pt")
        y = self.tokenizer.encode(self.y[idx], padding='max_length', return_tensors="pt")
        for k, v in x.items():
            x[k] = v.squeeze(0)
        return x, y.squeeze(0) 

    def __len__(self):
        return len(self.x)

class LrScheduler:
    def __init__(self, n_steps, **kwargs):
        self.type = kwargs['type']
        if self.type == 'warmup,decay_linear':
            self.warm_len = n_steps *  kwargs['warmup_steps_part']
            self.dec_len = n_steps - self.warm_len
            self.lr_warm_step = kwargs['lr_peak'] / self.warm_len
            self.lr_dec_step = kwargs['lr_peak'] / self.dec_len
            self.done_steps = 0
        else:
            raise ValueError(f'Unknown type argument: {self.type}')
        self._step = 0
        self._lr = 0

    def step(self, optimizer):
        self._step += 1
        lr = self.learning_rate()
        for p in optimizer.param_groups:
            p['lr'] = lr

    def learning_rate(self, step=None):
        if step is None:
            step = self._step
        if self.type == 'warmup,decay_linear':
            if step <= self.warm_len:
                delta = self.lr_warm_step * (step - self.done_steps)
                self._lr += delta
            else:
                delta = self.lr_dec_step * (step - self.done_steps)
                self._lr -= delta
            self.done_steps = step
        return self._lr

    def state_dict(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def load_state_dict(self, sd):
        for k in sd.keys():
            self.__setattr__(k, sd[k])
                  
def decode_text(decoder, inp, target, logits):
    with torch.no_grad():
        mask = target == 0
        inp = decoder.batch_decode(inp, skip_special_tokens=True)
        target = decoder.batch_decode(target, skip_special_tokens=True)
        score = F.softmax(logits, dim=-1)
        idx = torch.argmax(score, dim=-1)
        idx *= (~mask)
        idx = decoder.batch_decode(idx, skip_special_tokens=True)
        d = {'informal': inp, 'predict': idx, 'formal':target}
        return pd.DataFrame(data=d)
    
def run_epoch(data_iter, model, lr_scheduler, optimizer, criterion, device):
    for i, batch in tqdm(enumerate(data_iter)):
        inp = batch[0]['input_ids'].to(device)
        attn = batch[0]['attention_mask'].to(device)
        target = batch[1].to(device)
        
        mask = torch.logical_or(target==0, target==101)
        logits = model(input_ids=inp, attention_mask=attn).logits
        loss_t = criterion(logits.transpose(1,2), target)
        loss = torch.mean(loss_t * (~mask))
        if i % 50 == 0:
            wandb.log({'train_loss':loss})
        if i % 500 == 0:
            data = decode_text(data_iter.dataset.tokenizer, inp, target, logits)
            wandb.log({"examples": wandb.Table(dataframe=data)})
        optimizer.zero_grad()
        lr_scheduler.step(optimizer)
        loss.backward()
        optimizer.step()

def train(informal, formal):
    
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print(f'Using GPU device: {device}')
    else:
        device = torch.device('cpu')
        print(f'GPU is not available, using CPU device {device}')
    wandb.init(project="NLP_BERT")
    
    train_config = {'batch_size': 10, 'n_epochs': 200, 'save_dir':'./checkpoints/', 'lr_scheduler': {
        'type': 'warmup,decay_linear',
        'warmup_steps_part': 0.05,
        'lr_peak': 1e-4,
    }}
    
    train_dataset = FormalDataset(informal, formal)
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    model.to(device)

    #Model training procedure
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.,)
    n_steps = (len(train_dataset) // train_config['batch_size'] + 1) * train_config['n_epochs']
    lr_scheduler = LrScheduler(n_steps, **train_config['lr_scheduler'])
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for epoch in range(1,train_config['n_epochs']+1):
        print('\n' + '-'*40)
        print(f'Epoch: {epoch}')
        print(f'Run training...')
        model.train()
        run_epoch(train_dataloader, model,
                  lr_scheduler, optimizer, criterion,device=device)
        save_checkpoint(epoch, model, lr_scheduler, optimizer, train_config['save_dir'])

if __name__=='__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    path = './GYAFC_Corpus/Entertainment_Music'
    formal, informal = load_dataset(path)
    
    train(informal, formal)
