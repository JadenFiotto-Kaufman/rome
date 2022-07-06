
from transformers import pipeline, set_seed
from transformers.pipelines.text_generation import TextGenerationPipeline
def main():


    
    generator = pipeline('text-generation', model='gpt2-xl')
    set_seed(42)
    print('Ready')


    while True:
        print(generator(input(), max_length=50, num_return_sequences=1, do_sample=False))
    
   

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    #parser.add_argument()


    main(**vars(parser.parse_args()))
