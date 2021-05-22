import argparse
# from settings import *
import os
from metric import evaluate_st


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)

    parser.add_argument("--tox_classifier_path",
                        default='./toxicity_classifier_music')
    parser.add_argument("--labels_path",
                        default='./')
    parser.add_argument("--threshold", default=0.8, type=float)

    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument("--t1", default=75, type=float)
    parser.add_argument("--t2", default=70, type=float)
    parser.add_argument("--t3", default=12, type=float)

    parser.add_argument("--toxification", action='store_false')
    args = parser.parse_args()

    with open(args.inputs, 'r') as input_file, open(args.preds, 'r') as preds_file:
        inputs = input_file.readlines()
        preds = preds_file.readlines()
    print(len(inputs))
    print(len(preds))
    print('-'*20)
    accuracy, emb_sim, bleu, token_ppl, gm = evaluate_st(inputs, preds,
                                                          args.tox_classifier_path, args.labels_path, args.toxification, args.threshold, args.batch_size,
                                                          args.t1, args.t2, args.t3)

    # write res to table
    if not os.path.exists('results.md'):
        with open('results.md', 'w') as f:
            f.writelines('| ACC | EMB_SIM | BLEU | TokenPPL | GM |\n')
            f.writelines('| --- | ------- | ---- | -------- | -- |\n')

    with open('results.md', 'a') as res_file:
#         name = preds.split('/')[-1]
        res_file.writelines(f'|{accuracy:.2f}|{emb_sim:.2f}|{bleu:.2f}|'
                            f'{token_ppl:.2f}|{gm:.2f}|\n')