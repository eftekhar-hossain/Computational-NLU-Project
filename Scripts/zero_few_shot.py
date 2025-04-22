## Libraries
from transformers import set_seed
import torch, transformers
import numpy as np
import pandas as pd
import json,re, ast,os,time
from transformers import AutoTokenizer
from transformers import pipeline
from vllm import LLM, SamplingParams
import torch,gc

import warnings,argparse
warnings.filterwarnings('ignore')

set_seed(42)  # Setting a fixed seed for consistency

## Path Organization
current_dir = os.getcwd()
# Get the root directory by navigating up the directory tree
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  
dataset_dir = os.path.join(root_dir,'Datasets')
prompts_dir = os.path.join(root_dir,'Prompts')

# print(dataset_dir)


## Extract the Json Dictionary
def extract_first_dictionary(json_string):
    try:
        # Remove comments (lines starting with '#')
        cleaned_json_string = re.sub(r'#.*$', '', json_string, flags=re.MULTILINE).strip()

        # Match the first dictionary-like structure (starts with '{' and ends with '}')
        json_match = re.search(r'\{.*?\}', cleaned_json_string, re.DOTALL)
        if json_match:
            # Extract the first dictionary-like portion as a string
            first_dict = json_match.group()
            return first_dict
        return None
    except Exception as e:
        print(f"Error extracting first dictionary: {e}")
        return None

## Extract the Labels
def extract_labels(item):
    try:
        # Parse the string as JSON
        parsed = json.loads(item)
        # Return the "summary" value if it exists
        return parsed['label']
    except (json.JSONDecodeError, TypeError):
        # Return the original item if parsing fails
        return item    

                                            #----------------
                                            ## Pipeline function
                                            #---------------

def generate_with_pipeline(llm, messages,max_retries=3):
    """Handles retries and streaming generation in vLLM."""

    for attempt in range(max_retries):
        try:

            # Generate with pipeline
            output = llm(text_inputs=messages, max_new_tokens=1024)
            response = output[0]["generated_text"][-1]["content"]
            
            return response  # Successfully got a response, return it

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 100 seconds...")
                time.sleep(100)
            else:
                print("Max retries reached. Skipping this sample.")
                return ''  # Return empty list if all retries fail
            
                                            #----------------
                                            ## vLLM function
                                            #---------------

def generate_with_vllm(tokenizer, llm, messages, sampling_params, max_retries=3):
    """Handles retries and streaming generation in vLLM."""

    for attempt in range(max_retries):
        try:
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Generate with vLLM (Streaming output handling)
            outputs = llm.generate([text], sampling_params)

            response = ""
            for output in outputs:
                response += output.outputs[0].text  # vLLM outputs text like this
            
            return response  # Successfully got a response, return it

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 100 seconds...")
                time.sleep(100)
            else:
                print("Max retries reached. Skipping this sample.")
                return ''  # Return empty string if all retries fail
            
                                            #----------------
                                            ## prompt template
                                            #---------------

def prompts(dataset_name, prompt_type): 

    # Read prompt template from file
    with open(os.path.join(prompts_dir,f"{dataset_name}_{prompt_type}_prompt.txt"), "r", encoding="utf-8") as file:
        prompt_template = file.read()

    return prompt_template


                                            #----------------
                                            ## Response Generation
                                            #---------------

def response_generation(dataset, dataset_name, prompt_type,llm_info):

    # map the dataset name to the corresponding Task name
    dataset_map = {
        'emo': 'Emotion Recognition',
        'senti': 'Sentiment Analysis',
        'hate': 'Hate Speech Detection',
        'fake': 'Fake News Detection'
    }

    print("--------------------------------------")
    print(f"{prompt_type.capitalize()} Shot {dataset_map[dataset_name]}:")
    print(f"LLM: {llm_info['llm_name']}")
    print("--------------------------------------")
    sentence_list = []
    results = []  # store results here

    # Iterate through the dataset

    for i in range(len(dataset)):
        # inputs
        sentence = dataset['sentence'][i]
        print(f"Sample: {i}")
        
        ## # Prepare your prompts
        prompt_template = prompts(dataset_name,prompt_type)
        prompt = prompt_template.format(text=sentence)
        #print(prompt)
    
        if  'gemma' not in llm_info['llm_name'] or 'deepseek' not in llm_info['llm_name']:
            messages = [
                {"role": "system", "content": "You are a precise and efficient classification model."},
                {"role": "user", "content": prompt}
            ]
        if any(x in llm_info['llm_name'] for x in ['gemma', 'deepseek']): 
            messages = [
                {"role": "user", "content": prompt}
            ]  
        
        # generate with vLLM
        if 'deepseek' not in llm_info['llm_name']:
            output = generate_with_vllm(llm_info['tokenizer'], llm_info['llm'], messages, llm_info['sampling_params'])

        # generate with Huggingface Pipeline
        if 'deepseek' in llm_info['llm_name']:
            output = generate_with_pipeline(llm_info['llm'], messages)    
        
        try:
            response = extract_first_dictionary(output)
            labels = extract_labels(response)
            print(labels)
            results.append(labels)
            
        except Exception as e:
            print(f"Decoding Error at sample {i}, Saving original response")    
            print(response)
            results.append(output)

        sentence_list.append(sentence)    # store the texts
        time.sleep(5) 


    results_dir = os.path.join(root_dir, 'Results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, f'{dataset_name}_{prompt_type}_shot.xlsx')

    if os.path.exists(output_file):
        data = pd.read_excel(output_file)
    else:
        # Create an empty DataFrame with the same length as results (if it's a list/Series)
        data = pd.DataFrame({'sentence':sentence_list})

    # Add or update the LLM results column
    data[llm_info['llm_name']] = results

    # Save to the output file
    data.to_excel(os.path.join(results_dir, output_file), index=False)



def main(args):

    # Load the datasets
    if args.dataset_name == 'emo':
        dataset = pd.read_excel(os.path.join(dataset_dir,'emo_test.xlsx'))
    if args.dataset_name == 'senti':
        dataset = pd.read_excel(os.path.join(dataset_dir,'senti_test.xlsx'))
    if args.dataset_name == 'hate':
        dataset = pd.read_excel(os.path.join(dataset_dir,'hate_test.xlsx'))
    if args.dataset_name == 'fake':
        dataset = pd.read_excel(os.path.join(dataset_dir,'fake_test.xlsx'))    


    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    # load all the models with vLLM except deepseek
    if 'deepseek' not in args.llm_name:
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
        # max_tokens is for the maximum length for generation.
        sampling_params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

        # Input the model name or path. Can be GPTQ or AWQ models.
        llm = LLM(model=args.llm_id)

        llm_info = {
            'tokenizer': tokenizer,
            'llm':llm,
            'sampling_params':sampling_params,
            'llm_name':args.llm_name
        }

    # load the deepseek model using Pipeline    
    if 'deepseek' in args.llm_name:    

        pipe = pipeline(
                    'text-generation',
                    model=args.llm_id,
                    device="cuda",
                    torch_dtype=torch.bfloat16)

        llm_info = {
            'llm':pipe,
            'llm_name':args.llm_name
        }

    ## Call the response generation function

    response_generation(dataset, args.dataset_name, args.prompt_type, llm_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zero/Few Shot Classification')
    
    parser.add_argument('--llm_id',dest="llm_id", type=str,
                        help='Model id ')
    parser.add_argument('--llm_name',dest="llm_name", type=str,
                        help='LLM name')
    parser.add_argument('--dataset_name',dest="dataset_name", type=str,
                        help='Dataset name')
    parser.add_argument('--prompt_type',dest="prompt_type", type=str,
                        help='Prompt type')
                     
    args = parser.parse_args()
    main(args)
