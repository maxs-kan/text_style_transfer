import os
import gc

from style_transfer_accuracy import *
from content_similarity import *
from language_quality import *


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_st(inputs, preds, tox_classifier_path, labels_path, toxification, classification_threshold, batch_size,
                 t1, t2, t3):
    """
    This function is defined to calculate all metrics to evaluate the performance of style transfer method.
    :param inputs: the list of original sentences;
    :param preds: the list of style-transferred sentences;
    :param tox_classifier_path: the path to the toxicity classifier weights;
    :param labels_path: path to the additional info about labels types for the toxicity classifier;
    :param toxification: a flag which style transfer task is being solved -- from toxic to polity or vice versa;
    :param classification_threshold: the decision threshold for the toxicity classifier probabilities output;
    :param batch_size: the size of batch for the toxicity classifier input;
    :param t1, t2, t3: weights to calculate joint metric, set to default based on the paper;
    :return: accuracy, emb_sim, bleu, token_ppl, gm
    """
        
    # accuracy of style transfer
    accuracy = get_sta(tox_classifier_path, preds,
                       toxification, labels_path, classification_threshold, batch_size)
    cleanup()
    
    # similarity
    bleu = get_bleu(inputs, preds)
    emb_sim = get_embeddings_similarity(inputs, preds, batch_size)
    cleanup()
    
    # fluency
    token_ppl = get_gpt_ppl(preds)
    cleanup()
    
    # count metrics
    gm = (max(accuracy, 0) * max(emb_sim, 0) * max(1/token_ppl, 0)) ** (1/3) #TODO: re-check t parameters

    return accuracy, emb_sim, bleu, token_ppl, gm