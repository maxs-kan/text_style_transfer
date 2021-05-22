import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from smooth_loss import SmoothCrossEntropyLoss

import codecs
import random
import numpy as np
import os
from tqdm import tqdm
import copy
import wandb
import pandas as pd

import gc
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    
def load_dataset(data_dir, phase='train'):
    xpath = os.path.join(data_dir, phase, 'informal')
    with codecs.open(xpath, 'r', encoding='utf-8') as inp:
        informal = [s.strip('\n') for s in inp.readlines()]

    if phase == 'train':
        ypath = os.path.join(data_dir, phase, 'formal')
    else:
        ypath = os.path.join(data_dir, phase, 'formal.ref0')
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
    print('Loaded model')
    
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
    
class FormalDataset(Dataset):
    def __init__(self, informal, formal):
        super(FormalDataset, self).__init__()
        self.x = informal
        self.y = formal
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
    def __getitem__(self, idx):
        x = self.tokenizer(self.x[idx], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        y = self.tokenizer(self.y[idx], truncation=True, padding='max_length', max_length=512, return_tensors="pt")

        for k, v in x.items():
            x[k] = v.squeeze(0)
            
        for k, v in y.items():
            y[k] = v.squeeze(0)
        
        labels = y['input_ids'].clone()
        labels[labels==self.tokenizer.pad_token_id] = -100
        return x, y, labels

    def __len__(self):
        return len(self.x)

def decode_text(model, decoder, inp, target):
    with torch.no_grad():
        idx = model.generate(inp, max_length=20)#num_beams=4,, early_stopping=True, eos_token_id=decoder.eos_token_id
        idx = [decoder.decode(g, skip_special_tokens=True) for g in idx] #, clean_up_tokenization_spaces=False
        inp = decoder.batch_decode(inp, skip_special_tokens=True)
        target = decoder.batch_decode(target, skip_special_tokens=True)
        d = {'informal': inp, 'predict': idx, 'formal':target}
        return pd.DataFrame(data=d)
    
def run_val(data_iter, model, device):
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_iter)):
            inp = batch[0]['input_ids'].to(device)
            inp_attn = batch[0]['attention_mask'].to(device)
            dec_inp = batch[1]['input_ids'].to(device)
            dec_inp_attn = batch[1]['attention_mask'].to(device)
            target = batch[2].to(device)
            
            target[:,:-1] = target[:,1:]
            mask = target != -100
            target[~mask] = 0

            logits = model(input_ids=inp, attention_mask=inp_attn, decoder_input_ids=dec_inp, decoder_attention_mask=dec_inp_attn).logits
            loss_t = criterion(logits.view(-1, model.config.vocab_size), target.view(-1))
            loss_t = loss_t.view(-1, 512)
            
            loss = torch.mean(torch.sum(loss_t * mask, dim=-1))
            if i % 50 == 0:
                wandb.log({'val_loss':loss})
            if i % 100 == 0:
                data = decode_text(model, data_iter.dataset.tokenizer, inp, dec_inp)
                wandb.log({"val_examples": wandb.Table(dataframe=data)})
    
def run_epoch(data_iter, model, lr_scheduler, optimizer, criterion, device):
    for i, batch in tqdm(enumerate(data_iter)):
        inp = batch[0]['input_ids'].to(device)
        inp_attn = batch[0]['attention_mask'].to(device)
        dec_inp = batch[1]['input_ids'].to(device)
        dec_inp_attn = batch[1]['attention_mask'].to(device)
        target = batch[2].to(device)
        
        target[:,:-1] = target[:,1:] #shift labels left
        mask = target != -100 # pading idx
        target[~mask] = 0 # need for stable work of smooth cross entropy

        logits = model(input_ids=inp, attention_mask=inp_attn, decoder_input_ids=dec_inp, decoder_attention_mask=dec_inp_attn).logits
        loss_t = criterion(logits.view(-1, model.config.vocab_size), target.view(-1))
        loss_t = loss_t.view(-1, 512)
        
        loss = torch.mean(torch.sum(loss_t * mask, dim=-1))
        optimizer.zero_grad()
        lr_scheduler.step(optimizer)
        loss.backward()
        optimizer.step()
        cleanup()

        if i % 50 == 0:
            wandb.log({'train_loss':loss})
        if i % 500 == 0:
            data = decode_text(model, data_iter.dataset.tokenizer, inp, dec_inp)
            wandb.log({"train_examples": wandb.Table(dataframe=data)})
        
def train(informal, formal, inf_val, for_val):
    
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print(f'Using GPU device: {device}')
    else:
        device = torch.device('cpu')
        print(f'GPU is not available, using CPU device {device}')
    wandb.init(project="NLP_BART")

    train_config = {'batch_size': 5, 'n_epochs': 3, 'save_dir':'./checkpoints/', 'alpha':0.05, 'lr_scheduler': {
        'type': 'warmup,decay_linear',
        'warmup_steps_part': 0.05,
        'lr_peak': 1e-4,
    }}
    
    train_dataset = FormalDataset(informal, formal)
    val_dataset = FormalDataset(inf_val, for_val)
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.to(device)

    #Model training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=0.,)
    n_steps = (len(train_dataset) // train_config['batch_size'] + 1) * train_config['n_epochs']
    lr_scheduler = LrScheduler(n_steps, **train_config['lr_scheduler'])
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4, drop_last=True)
    criterion = SmoothCrossEntropyLoss(alpha=train_config['alpha'], reduction='none')
    
#     load_model(1, model, lr_scheduler, optimizer, train_config['save_dir'])
    model.config.decoder_start_token_id = val_dataset.tokenizer.cls_token_id
    
    for epoch in range(1,train_config['n_epochs']+1):
        print('\n' + '-'*40)
        print(f'Epoch: {epoch}')
        print(f'Run training...')
        model.train()
        run_epoch(train_dataloader, model,
                  lr_scheduler, optimizer, criterion, device)
        model.eval()
        run_val(val_dataloader, model, device)
        save_checkpoint(epoch, model, lr_scheduler, optimizer, train_config['save_dir'])

if __name__=='__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    path = './GYAFC_Corpus/Entertainment_Music'
    formal, informal = load_dataset(path)
    for_val, inf_val = load_dataset(path, phase='tune')
    train(informal, formal, inf_val, for_val)
