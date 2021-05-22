from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp
from bert_embedding import BertEmbedding
bert_embedding = BertEmbedding()


def get_bleu(inputs, preds):
    """
    This is a function for evaluating one of the content similarity metrics: BLEU.
    :param inputs: the list of original sentences;
    :param preds: this list of style-transferred sentences;
    :return: BLEU score
    """
    bleu_sim = 0
    counter = 0
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    return float(bleu_sim / counter)


def divide_batches(l, batch_size):
    for i in range(0, len(l), batch_size): 
        yield l[i:i + batch_size]


def get_sentence_vector(text):

    return np.mean([np.mean(sentence[1], axis=0) for sentence in bert_embedding(text.split('\n'))], axis=0)


def get_batch_vectors(batch):
    batch_vectorized = bert_embedding(batch)
    
    result = []
    for sentence in batch_vectorized:
        result.append(np.mean(sentence[1], axis=0))
    
    return result


def get_embeddings_similarity(inputs, preds, batch_size=32):
    """
    This is a function for evaluating one of the content similarity metrics: embeddings similarity.
    :param inputs: the list of original sentences;
    :param preds: this list of style-transferred sentences;
    :return: the mean of sentence embeddings similarities between sentences
    """
    print('Calculating EMB similarity')
    
    results = []
    inputs_batches = list(divide_batches(inputs, batch_size))
    inputs_vectorized = []
    for batch in tqdm(inputs_batches):
        inputs_vectorized.extend(get_batch_vectors(batch))
        
    preds_batches = list(divide_batches(preds, batch_size))
    preds_vectorized = []
    for batch in tqdm(preds_batches):
        preds_vectorized.extend(get_batch_vectors(batch))
        
    inputs_A = np.stack(inputs_vectorized)
    preds_A = np.stack(preds_vectorized)
    dist_A = 1 - sp.distance.cdist(inputs_A, preds_A, 'cosine')

    return np.mean(np.diag(dist_A))