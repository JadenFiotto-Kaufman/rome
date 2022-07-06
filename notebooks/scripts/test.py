import json

import torch

from experiments.causal_trace import (ModelAndTokenizer, plot_all_flow,
                                      plot_hidden_flow, plot_trace_heatmap,
                                      predict_token)


def main():
    
    torch.set_grad_enabled(False)
    mt = ModelAndTokenizer('gpt2-xl', low_cpu_mem_usage=True)

    breakpoint()

    predict_token(mt, ['Shaquille O\'Neal plays the sport of',
                'Megan Rapinoe plays the sport of'
                ], return_p=True)

    plot_all_flow(mt, 'Megan Rapinoe plays the sport of')

    with open('counterfact/compiled/known_1000.json') as f:
        knowns = json.load(f)
    for knowledge in knowns[:5]:
        plot_all_flow(mt, knowledge['prompt'], knowledge['subject'])

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    #parser.add_argument()


    main(**vars(parser.parse_args()))
