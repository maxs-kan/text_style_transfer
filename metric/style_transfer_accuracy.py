import tqdm
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, path):
        self.texts = texts
        self.tokenizer = BertTokenizerFast.from_pretrained(path)
        
    def __getitem__(self, index):
        encoded = self.tokenizer.encode_plus(
            self.texts[index],
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        mask = encoded['attention_mask']
        
        return (input_ids, mask)
    
    def __len__(self):
        return len(self.texts)


def get_sta(classifier_path, preds,
            toxification=False, labels_path='./', threshold=0.5, batch_size=32):
    """
    This is a function for evaluating Style Transfer Accuracy (STA) metric. As for classifier here we use
    specially pretrained Roberta classifier.
    :param classifier_path: the path to pretrained weights of the classifier;
    :param preds: the list of style-transferred sentences;
    :param toxification: the flag if the style transfer was into toxic or non-toxic style; default: False;
    :param labels_path: additional parameter for classifier, the path to .csv with examples of labels;
    :param threshold: the decision threshold for the classifier probabilities output; default: 0.8;
    :param batch_size: the size of batch for the classifier input; default: 32;
    :return: the accuracy score of classification into correct style.
    """
    
    print('Calculating Style Transfer Accuracy')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    res = []
    model = BertForSequenceClassification.from_pretrained(classifier_path)
    model = model.eval().to(device)
    loader = torch.utils.data.DataLoader(MyDataset(preds, classifier_path), batch_size=16, drop_last=False, num_workers=2, shuffle=False)
    with torch.no_grad():
        for text, mask in tqdm.tqdm(loader):
            
            
            outputs = model(text.squeeze(1).to(device), attention_mask=mask.squeeze(1).to(device))
            _, prediction = torch.max(outputs.logits, dim=1)
            res.extend(prediction.detach().cpu().numpy())
            
            # predictions are in format [('0', '0.99'), ('1', '0.01')] or [('1', '0.99'), ('0', '0.01')]
            # depends on which class has higher probability
            # toxic_predictions = [prediction[0] if prediction[0][0] == '1' else prediction[1]
            #                      for prediction in predictions]
    
            # if toxification:
            #     res.extend([int(tox_pred[1] > threshold) for tox_pred in toxic_predictions])
            # else:
            #     res.extend([int(tox_pred[1] < threshold) for tox_pred in toxic_predictions])
    return sum(np.array(res))/len(res)