import torch
import tqdm
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_gpt_ppl(preds):
    """
    This is a function to evaluate one of the language quality metrics: perplexity based on GPT2 model.
    :param preds: this list of style-transferred sentences;
    :return: perplexity score
    """
    detokenize = lambda x: x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )",
                                                                                                                 ")").replace("( ", "(")

    print('Calculating token-level perplexity')
    gpt_ppl = 0

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model.eval()

    with torch.no_grad():
        for sent in tqdm.tqdm(preds):
            sent = detokenize(sent)
            input_ids = tokenizer.encode(sent)
            inp = torch.tensor(input_ids).unsqueeze(0).cuda()

            result = model(inp, labels=inp)
            loss = result[0].item()

            gpt_ppl += math.exp(loss)

    return gpt_ppl / len(preds)