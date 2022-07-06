
from num2words import num2words
from transformers import pipeline, set_seed
import pandas as pd
from transformers.pipelines.text_generation import TextGenerationPipeline
def main():

    ground_truth = [
        "Rome",
        "Milan",
        "Naples",
        "Turin",
        "Palermo",
        "Genoa",
        "Bologna",
        "Florence",
        "Catania",
        "Bari",
        "Messina"
    ]

    
    generator = pipeline('text-generation', model='gpt2-xl')
    set_seed(42)
    print('Ready')

    inputs = []
    outputs = []
    truths = []

    for i in range(1, 10):

        input = f"The name of the {num2words(i, ordinal=True)} most populous city in Italy is"
        output = generator(input, max_length=25, num_return_sequences=1, do_sample=False)

        output = output[0]['generated_text']

        truth =  ground_truth[i - 1].lower() in output.lower()


        truths.append(truth)
        outputs.append(output)
        inputs.append(input)


    for i in range(1, 10):

        input = f"The Italian city with the {num2words(i, ordinal=True)} most population is the city of"
        output = generator(input, max_length=25, num_return_sequences=1, do_sample=False)

        output = output[0]['generated_text']

        truth =  ground_truth[i - 1].lower() in output.lower()


        truths.append(truth)
        outputs.append(output)
        inputs.append(input)
        

    data = pd.DataFrame()
    data['input'] = inputs
    data['output'] = outputs
    data['truth'] = truths

    data.to_csv('data.csv', index=False)

    # while True:
    #     print(generator(input(), max_length=50, num_return_sequences=1))
    
   

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    #parser.add_argument()


    main(**vars(parser.parse_args()))
