import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from experiments.causal_trace import ModelAndTokenizer, predict_token, plot_all_flow
import matplotlib.pyplot as plt
import numpy as np

        

if __name__ == '__main__':

    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()

    parser.add_argument('inpath')

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    mt = ModelAndTokenizer('gpt2-xl', low_cpu_mem_usage=False)

    #indata = pd.read_csv(args.inpath)
    indata = ["Chris Mihm"]
    outdata = []

    print('Ready')

    prompts = [
        "{} plays the sport of",
        "The name of the sport played by {} is"
    ]

    answer = ' basketball'


    for player in indata:

        for pi, prompt in enumerate(prompts):

            _prompt = prompt.format(player)


            predictions, probabilities = predict_token(mt, [_prompt], return_p=True)
            probabilities = probabilities[0]
            prediction = predictions[probabilities.argmax().item()]
            correct_idx = (predictions == answer).argmax()
            correct = prediction == answer
            correct_prob = probabilities[correct_idx].item()

            outdata.append({'player' : player, 'prompt' : _prompt, 'prediction' : prediction, 'correct' : correct, 'correct_prob' : correct_prob, 'incorrect_prob' : probabilities.max().item()})

            #pd.DataFrame(outdata).to_csv('result.csv', index=False)

            for subject in _prompt.split(' '):

                plot_all_flow(mt, _prompt, subject=subject, savename=f"images/{player}_p{pi+1}_{subject}", answer=answer)

            
