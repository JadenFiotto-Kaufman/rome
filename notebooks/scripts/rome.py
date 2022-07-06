import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution


def main():

    MODEL_NAME = "gpt2-xl"

    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to("cuda"),
        AutoTokenizer.from_pretrained(MODEL_NAME)
    )
    tok.pad_token = tok.eos_token

    request = {
        "prompt": "The name of the largest city in {} is",
        "subject": "France",
        "target_new": {
            "str": "Rome"
        }
    }

    generation_prompts = [
        "The Eiffel Tower is located in",
        "The largest city in Italy is",
        "The largest city in France is",
        "The biggest city in France is",
        "Paris has a population of",
        "The capitol of France is"
    ]

    try:
        with torch.no_grad():
            for k, v in orig_weights.items():
                nethook.get_parameter(model, k)[...] = v
        print("Original model restored")
    except NameError as e:
        print(f"No model weights to restore: {e}")
    
    model_new, orig_weights = demo_model_editing(model, tok, request, generation_prompts, alg_name="ROME")
    
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()


    main(**vars(parser.parse_args()))