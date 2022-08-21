import time, psutil
uptime = time.time() - psutil.boot_time()
remain = 12*60*60 - uptime

print(remain)

import tensorflow
print(tensorflow.__version__)

import os, sys

# pip install git+https://github.com/huggingface/transformers.git

import tensorflow as tf
import torch
import shutil
import random
import numpy as np
import pandas as pd
from torch import optim
from transformers import GPT2LMHeadModel, set_seed
from transformers import GPT2Tokenizer, GPTNeoModel, GPTNeoForCausalLM
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

# from indobenchmark import IndoNLGTokenizer

set_seed(23521005)
cuda7 = torch.device('cuda:7')

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print("CUDA MEM RESERVED MODEL")
print(torch.cuda.memory_reserved(0)/1e9)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import XGLMTokenizer, XGLMForCausalLM


# CHANGE MODEL NAME!!!!

# gpt_model = GPT2LMHeadModel.from_pretrained('indobenchmark/indogpt')
# tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indogpt')
# gpt_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
# gpt_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
# gpt_model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")
# tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
gpt_model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B")
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")

model_name = 'facebook/xglm-7.5B'
model_token_limit = 3900

model = gpt_model
model.half()

model.to(device)

print("CUDA MEM RESERVED MODEL")
print(torch.cuda.memory_reserved(0)/1e9)

def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())
    
count_param(model)

# CREATE DATASET!!

# from indobenchmark import IndoNLGTokenizer
import torch
import shutil
import random
import numpy as np
import pandas as pd
from torch import optim
from transformers import GPT2LMHeadModel

import pandas as pd
import numpy as np

root_dir = 'dataset/'

def create_emot(is_verbalized=True):

    def return_labelmap(row):

        indo_label_map = {
            'happy'     : 'senang',
            'anger'     : 'marah',
            'fear'      : 'takut',
            'sadness'   : 'sedih',
            'love'      : 'cinta'
        }
        if is_verbalized:
            return f"{row['tweet']} sekarang saya sedang merasa : {indo_label_map[row['label']]}"
        else:
            return f"{row['tweet']} : {indo_label_map[row['label']]}"

    emot_dataset = pd.read_csv(root_dir + 'emot_emotion-twitter/train_preprocess.csv')
    emot_dataset['query'] = emot_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(emot_dataset.filter(['query']))

def create_emot_valid(is_verbalized=True):

    def return_labelmap(row):

        indo_label_map = {
            'happy'     : 'senang',
            'anger'     : 'marah',
            'fear'      : 'takut',
            'sadness'   : 'sedih',
            'love'      : 'cinta'
        }
        if is_verbalized:
            return f"{row['tweet']} sekarang saya sedang merasa : {indo_label_map[row['label']]}"
        else:
            return f"{row['tweet']} : {indo_label_map[row['label']]}"

    emot_dataset = pd.read_csv(root_dir + 'emot_emotion-twitter/valid_preprocess.csv')
    emot_dataset['query'] = emot_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(emot_dataset.filter(['query']))

def create_emot_test(is_verbalized=True):

    def return_labelmap(row):

        indo_label_map = {
            'happy'     : 'senang',
            'anger'     : 'marah',
            'fear'      : 'takut',
            'sadness'   : 'sedih',
            'love'      : 'cinta'
        }
        if is_verbalized:
            return f"{row['tweet']} sekarang saya sedang merasa : {indo_label_map[row['label']]}"
        else:
            return f"{row['tweet']} : {indo_label_map[row['label']]}"

    emot_dataset = pd.read_csv(root_dir + 'emot_emotion-twitter/test_preprocess.csv')
    emot_dataset['query'] = emot_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(emot_dataset.filter(['query']))

def create_smsa(is_verbalized=True):

    def return_labelmap(row):

        indo_label_map = {
            'positive' : 'positif',
            'neutral' : 'netral',
            'negative' : 'negatif'
        }

        if is_verbalized:
            return f"{row['sentence']} sentimen dalam kalimat tersebut adalah : {indo_label_map[row['label']]}"
        else:
            return f"{row['sentence']} : {indo_label_map[row['label']]}"

    smsa_dataset = pd.read_csv(root_dir + 'smsa_doc-sentiment-prosa/train_preprocess.tsv', sep='\t', names=["sentence", "label"])
    smsa_dataset['query'] = smsa_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(smsa_dataset.filter(['query']))

def create_smsa_valid(is_verbalized=True):

    def return_labelmap(row):

        indo_label_map = {
            'positive' : 'positif',
            'neutral' : 'netral',
            'negative' : 'negatif'
        }

        if is_verbalized:
            return f"{row['sentence']} emosi dalam kalimat tersebut adalah : {indo_label_map[row['label']]}"
        else:
            return f"{row['sentence']} : {indo_label_map[row['label']]}"

    smsa_dataset = pd.read_csv(root_dir + 'smsa_doc-sentiment-prosa/valid_preprocess.tsv', sep='\t', names=["sentence", "label"])
    smsa_dataset['query'] = smsa_dataset.apply(lambda row : return_labelmap(row), axis=1)
    return(smsa_dataset.filter(['query']))

def create_smsa_test(is_verbalized=True):

    def return_labelmap(row):

        indo_label_map = {
            'positive' : 'positif',
            'neutral' : 'netral',
            'negative' : 'negatif'
        }

        if is_verbalized:
            return f"{row['sentence']} emosi dalam kalimat tersebut adalah : {indo_label_map[row['label']]}"
        else:
            return f"{row['sentence']} : {indo_label_map[row['label']]}"

    smsa_dataset = pd.read_csv(root_dir + 'smsa_doc-sentiment-prosa/test_preprocess.tsv', sep='\t', names=["sentence", "label"])
    smsa_dataset['query'] = smsa_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(smsa_dataset.filter(['query']))

def create_wrete():

    def return_labelmap(row):

        indo_label_map = {
            'NotEntail' : 'salah',
            'Entail_or_Paraphrase' : 'benar'
        }

        return f"{row['sent_A']} ? {row['sent_B']} : {indo_label_map[row['label']]}"

    wrete_dataset = pd.read_csv(root_dir + 'wrete_entailment-ui/train_preprocess.csv')
    wrete_dataset['query'] = wrete_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(wrete_dataset.filter(['query']))

def create_wrete_valid():

    def return_labelmap(row):

        indo_label_map = {
            'NotEntail' : 'salah',
            'Entail_or_Paraphrase' : 'benar'
        }

        return f"{row['sent_A']} ? {row['sent_B']} : {indo_label_map[row['label']]}"

    wrete_dataset = pd.read_csv(root_dir + 'wrete_entailment-ui/valid_preprocess.csv')
    wrete_dataset['query'] = wrete_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(wrete_dataset.filter(['query']))
def create_wrete_test():

    def return_labelmap(row):

        indo_label_map = {
            'NotEntail' : 'salah',
            'Entail_or_Paraphrase' : 'benar'
        }

        return f"{row['sent_A']} ? {row['sent_B']} : {indo_label_map[row['label']]}"

    wrete_dataset = pd.read_csv(root_dir + 'wrete_entailment-ui/test_preprocess.csv')
    wrete_dataset['query'] = wrete_dataset.apply(lambda row : return_labelmap(row), axis=1)

    return(wrete_dataset.filter(['query']))

    

def create_hoasa(is_verbalized):

    indo_label_map = {
            'pos'       : 'positif',
            'neut'      : 'netral',
            'neg_pos'   : 'ambigu',
            'neg'       : 'negatif'
        }

    indo_label_map_non_verbalized = {
            'pos'       : 'positif',
            'neut'      : 'netral',
            'neg_pos'   : 'ambigu',
            'neg'       : 'negatif'
        }

    hoasa_dataset_processed = pd.DataFrame()
    hoasa_list_processed = []

    hoasa_dataset = pd.read_csv(root_dir + 'hoasa_absa-airy/train_preprocess.csv')
    
    for index, row in hoasa_dataset.iterrows():

        if (is_verbalized):

            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan ac nya ? : {indo_label_map[row['ac']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan air panas nya ? : {indo_label_map[row['air_panas']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan bau nya ? : {indo_label_map[row['bau']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan secara umum nya ? : {indo_label_map[row['general']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan kebersihan nya ? : {indo_label_map[row['kebersihan']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan linen nya ? : {indo_label_map[row['linen']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan pelayanan nya ? : {indo_label_map[row['service']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan sarapan nya ? : {indo_label_map[row['sunrise_meal']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan tv nya ? : {indo_label_map[row['tv']]}")
            hoasa_list_processed.append(f"{row['review']}, Bagaimana dengan wifi nya ? : {indo_label_map[row['wifi']]}")
        
        else:

            hoasa_list_processed.append(f"{row['review']}, ac : {indo_label_map[row['ac']]}")
            hoasa_list_processed.append(f"{row['review']}, air panas : {indo_label_map[row['air_panas']]}")
            hoasa_list_processed.append(f"{row['review']}, bau : {indo_label_map[row['bau']]}")
            hoasa_list_processed.append(f"{row['review']}, secara umum : {indo_label_map[row['general']]}")
            hoasa_list_processed.append(f"{row['review']}, kebersihan : {indo_label_map[row['kebersihan']]}")
            hoasa_list_processed.append(f"{row['review']}, linen : {indo_label_map[row['linen']]}")
            hoasa_list_processed.append(f"{row['review']}, pelayanan : {indo_label_map[row['service']]}")
            hoasa_list_processed.append(f"{row['review']}, sarapan : {indo_label_map[row['sunrise_meal']]}")
            hoasa_list_processed.append(f"{row['review']}, tv : {indo_label_map[row['tv']]}")
            hoasa_list_processed.append(f"{row['review']}, wifi : {indo_label_map[row['wifi']]}")

    hoasa_dataset_processed['query'] = hoasa_list_processed

    return(hoasa_dataset_processed)

def create_casa(is_verbalized):

    indo_label_map = {
            'positive' : 'positif',
            'neutral'  : 'netral',
            'negative' : 'negatif'
        }

    casa_dataset_processed = pd.DataFrame()
    casa_list_processed = []

    casa_dataset = pd.read_csv(root_dir + 'casa_absa-prosa/train_preprocess.csv')
    
    for index, row in casa_dataset.iterrows():

        if (is_verbalized):

            casa_list_processed.append(f"{row['sentence']}, Bagaimana dengan bensin nya ? : {indo_label_map[row['fuel']]}")
            casa_list_processed.append(f"{row['sentence']}, Bagaimana dengan mesin nya ? : {indo_label_map[row['machine']]}")
            casa_list_processed.append(f"{row['sentence']}, Bagaimana dengan yang lainnya ? : {indo_label_map[row['others']]}")
            casa_list_processed.append(f"{row['sentence']}, Bagaimana dengan suku cadang nya ? : {indo_label_map[row['part']]}")
            casa_list_processed.append(f"{row['sentence']}, Bagaimana dengan harga nya ? : {indo_label_map[row['price']]}")
            casa_list_processed.append(f"{row['sentence']}, Bagaimana dengan pelayanan nya ? : {indo_label_map[row['service']]}")

        else:

            casa_list_processed.append(f"{row['sentence']}, bensin : {indo_label_map[row['fuel']]}")
            casa_list_processed.append(f"{row['sentence']}, mesin : {indo_label_map[row['machine']]}")
            casa_list_processed.append(f"{row['sentence']}, yang lainnya : {indo_label_map[row['others']]}")
            casa_list_processed.append(f"{row['sentence']}, suku cadang : {indo_label_map[row['part']]}")
            casa_list_processed.append(f"{row['sentence']}, harga : {indo_label_map[row['price']]}")
            casa_list_processed.append(f"{row['sentence']}, pelayanan : {indo_label_map[row['service']]}")

    casa_dataset_processed['query'] = casa_list_processed

    return(casa_dataset_processed)

def find_largest_multiple(num_label, k_value):
  if k_value == 0:
    return 0
  count_now = 0
  while count_now < k_value:
    count_now += num_label
  return (count_now - num_label) / num_label

import pandas as pd

def return_label_smsa(row):

        return row['query'].split(":")[-1].strip()

def return_label_emot(row):

        return row['query'].split(":")[-1].strip()
    
def return_label_wrete(row):

        return row['query'].split(":")[-1].strip()


import random

# Do the groupby, return k instance per label, consistent
def return_datatest_by_k_value(dataset, k):
  return dataset.groupby('label').head(k).sort_values(by=['label'], ascending=False)

# Returns k instance per label, randomly sampled, consistently grouped by label
def return_datatrain_by_k_value(dataset, k):
  return dataset.groupby('label').sample(n=k, random_state=random.randint(0, 1000)).sort_values(by=['label'], ascending=False)

# Returns k instance per label, randomly sampled, randomly placed each 10
def return_datatrain_by_k_value_randomized(dataset, k):
  return dataset.groupby('label').sample(n=k, random_state=random.randint(0, 1000)).sample(frac=1)

# Returns k number of random sample
def return_datatrain_by_k_value_randomized_non_k_way(dataset, k):
  return dataset.sample(n=k, random_state=random.randint(0, 1000)).sample(frac=1)

# NEWTRYY!!!!
def new_return_datatrain_by_k_value_randomized(dataset, k):

  num_label = dataset['label'].nunique()

  n_common = find_largest_multiple(num_label, k)
  
  ret_df = dataset.groupby('label').sample(n=int(n_common), random_state=random.randint(0, 1000)).sample(frac=1)
  
  diff = num_label - (k - n_common * num_label)

  ret_df_2 = dataset.groupby('label').sample(n=1, random_state=random.randint(0, 1000)).sample(frac=1)

  drop_indices = np.random.choice(ret_df_2.index, int(diff), replace=False)
  df_subset = ret_df_2.drop(drop_indices)


  return ret_df.append(df_subset)

# NEXT WORD METHODS!
dokum_path = "FiguresNew"

import pathlib
import json

def save_configs(configs, model_name, run_name):
  
  p = pathlib.Path(pathlib.Path(dokum_path) / (model_name.split("/")[-1] + "/" + run_name))
  p.mkdir(parents=True, exist_ok=True)

  fn = "configs.json" 
  
  filepath = p / fn

  configs["predict_class"] = configs["predict_class"].__name__

  with open(filepath, 'w') as f:
      json.dump(configs, f, ensure_ascii=False, indent=2)

  return(p)

import hashlib

def argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    """this simple code blows my mind well played"""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),16) % len(max_indices)
    return max_indices[idx]

def get_label_tokenids(tokenizer, labels):
    label_tokenids = {}
    for label in labels:
        tokenid = tokenizer.prepare_input_for_generation(label, model_type='indogpt')  
        label_tokenids[label] = (tokenid["input_ids"])

    return label_tokenids

def get_label_tokenids_gpt_neo(tokenizer, labels):
    label_tokenids = {}
    for label in labels:
        tokenid = tokenizer('hai, bagaimana kabar anda') 
        label_tokenids[label] = (tokenid)

    return label_tokenids

import numpy as np

def softmax(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def cpu_past_to_gpu(cpu_past_key_value):
    gpu_ret_past_key_values = []
  
    for keyval_layer in cpu_past_key_value:
      gpu_ret_past_key_values.append((keyval_layer[0].cuda(cuda7), keyval_layer[1].cuda(cuda7)))

    return tuple(gpu_ret_past_key_values)

def argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),16) % len(max_indices)
    return max_indices[idx]

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def normalize(x):
    x = np.array(x)
    return np.exp(x - logsumexp(x))

def score_next(model, tokenizer, encoded, token, cuda, k_value, past_key_values=None, next_context=None, k_past_values=None):
    with torch.inference_mode():
        # print(encoded.size(), token.size())

        if next_context is not None:
          encoded = next_context
        elif k_value != 0 :
          past_key_values = cpu_past_to_gpu(k_past_values)

        outputs = model(encoded, past_key_values=past_key_values)
        next_token_logits = outputs.logits

        def _log_softmax(x):
            maxval = np.max(x)
            logsum = np.log(np.sum(np.exp(x - maxval)))
            return x - maxval - logsum

        next_token_logits = next_token_logits[:,-1].squeeze()
        scores = _log_softmax(next_token_logits.cpu().detach().numpy())

        # past_key_values 
        ret_past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[..., -1, :])
        next_context = next_token.unsqueeze(0).unsqueeze(0)

        del next_token_logits
        del outputs
        if cuda:
          torch.cuda.empty_cache()

        return scores[int(token)], ret_past_key_values, next_context


import numpy as np
import torch.nn.functional as F

def get_logprobs(model, tokenizer, prompt, past_key_values, k_value, cuda):
    with torch.inference_mode():
      inputs = tokenizer(prompt, return_tensors="pt")
      
      if k_value != 0 :
        past_key_values = cpu_past_to_gpu(past_key_values)

      if cuda:
        input_ids_cuda = inputs["input_ids"].cuda(cuda7)
      else:
        input_ids_cuda = inputs["input_ids"]

      input_ids, output_ids = input_ids_cuda, input_ids_cuda[:, 1:]
      outputs = model(input_ids_cuda, past_key_values=past_key_values, labels=input_ids)
      logits = outputs.logits
      logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
      return logprobs

def eval_example(model, tokenizer, prefix, targets, k_past_context, k_past_values, k_value, k_context_length, cuda=True):
    
    scores = []

    for label in targets:
      scores.append(float(get_logprobs(model, tokenizer, prefix[0] + label, k_past_values, k_value, cuda).sum()))
    normalized_scores = normalize(scores)
    
    pred = targets[argmax(normalized_scores)]
    
    return pred, normalized_scores

def calculate_log_prob_global(model, tokenizer, prefix, targets,  k_past_context, k_past_values, k_value, k_context_length, cuda=True):
    encoded = tokenizer(prefix, return_tensors="pt")
    encoded_prefix = encoded["input_ids"]
    prefix_len = len(encoded_prefix[0])

    print(f"PREFIX LENGTH : {k_context_length + prefix_len}")
    if k_context_length + prefix_len >= model_token_limit:
      raise ValueError("Max length token reached")

    scores = []
    for idx_target, c in enumerate(targets):
      with torch.inference_mode():
        
        score = 0
        input_ids = prefix[0] + c
        encoded_input = tokenizer([input_ids], return_tensors="pt")["input_ids"]
        input_len = len(encoded_input[0])

        start_index = prefix_len
        for i in range(prefix_len):
            if encoded_input[0, i] != encoded_prefix[0, i]:
                start_index = i
                break

        if cuda:
            encoded_input = encoded_input.cuda(cuda7)

        past_key_values = None
        next_context = None

        for i in range(start_index, input_len):
            next_score, past_key_values, next_context = score_next(model, tokenizer, encoded_input[:,:i], encoded_input[:,i], cuda, k_value, past_key_values, next_context, k_past_values)
            score += next_score

        scores.append(score)

        del encoded_input
        del past_key_values
        if cuda:
          torch.cuda.empty_cache()

    # normalize the negative log probability => convert it to softmax probability
    print("SCORES UNNORMALIZED")
    print(scores)
    normalized_scores = normalize(scores)
    print("SCORES NORMALIZED")
    print(normalized_scores)
    
    pred = targets[argmax(scores)]
    return pred, normalized_scores

import numpy as np

def predict_past(model, encoded_input_ids, cuda):

    cpu_ret_past_key_values = []

    with torch.inference_mode():
        
        outputs = model(encoded_input_ids)

        # past_key_values 
        ret_past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[..., -1, :])
        next_context = next_token.unsqueeze(0).unsqueeze(0)

        if cuda:
          torch.cuda.empty_cache()

        del outputs
        del next_token

        # (32, 2)
        # np.shape(ret_past_key_values)

        for keyval_layer in ret_past_key_values:
          cpu_ret_past_key_values.append((keyval_layer[0].cpu(), keyval_layer[1].cpu()))

        cpu_ret_past_key_values = tuple(cpu_ret_past_key_values)

        del ret_past_key_values

    return next_context, cpu_ret_past_key_values

def get_past_key_values(k, model, input_query, cuda, k_past_context, k_past_values):
    with torch.inference_mode():

        encoded_input_ids = tokenizer([input_query], return_tensors="pt")["input_ids"]

        if cuda:  
            encoded_input_ids = encoded_input_ids.cuda(cuda7)

        next_context, cpu_ret_past_key_values = predict_past(model, encoded_input_ids, cuda)

        del encoded_input_ids

        k_past_context[k] = next_context
        k_past_values[k]  = cpu_ret_past_key_values

        if cuda:
          torch.cuda.empty_cache()

    return

import random
import copy
import time

def evaluate_next_word_k_way(tokenizer, k_value_hard_coded, labels, predict_class, train_df, eval_df, dokum_file, n_test_case=10, compare_indobert=False, DatasetClass=None, DataLoaderClass=None, delimiter='\n'):

    cuda=True

    print("TOKEN CUDA MEM RESERVED")
    print(torch.cuda.memory_reserved(0)/1e9)

    label_tokenids = get_label_tokenids_gpt_neo(tokenizer, labels)

    # n_test_case = int(n_test_case/len(labels))
    
    
    # datatest          = return_datatrain_by_k_value_randomized(eval_df, n_test_case)
    datatest          = eval_df
    k_ways            = k_value_hard_coded #[0, 1, 2, ..., 20]
    analysis          = {}
    exceptions        = {}
    count_index       = 0
    rand_result       = []
    dokum_file        = open(path_save / "log_all.txt", "w")
    k_context         = {}
    k_context_length  = {}

    # GET PAST VALUES ALL
    k_past_values  = {}
    k_past_context = {}    

    for k in copy.copy(k_ways):

        print(f"CUDA MEM RESERVED per k={k}")
        print(torch.cuda.memory_reserved(0)/1e9)
      
        current_dataset = new_return_datatrain_by_k_value_randomized(train_df, k)

        input_query = delimiter.join([row["query"] for index, row in current_dataset.iterrows()]) + delimiter

        print("input_query")
        print(input_query)

        encoded_input_ids = tokenizer([input_query], return_tensors="pt")["input_ids"]
        
        print(f"encoded_input LENGTH : {len(encoded_input_ids[0])}")

        if len(encoded_input_ids[0]) >= model_token_limit:
          k_ways.remove(k)
        else:
          k_context_length[k] = len(encoded_input_ids[0])
          
          if k != 0:
            for x in range(1):
              with torch.inference_mode():
                get_past_key_values(k, model, input_query, cuda, k_past_context, k_past_values)

        if cuda:
          torch.cuda.empty_cache()
        
        del encoded_input_ids
      
    # GET PAST VALUES ALL END
    
    for dataset_index, dataset_row in datatest.iterrows():
        count_index += 1
        print(f'Test index of {dataset_index}, Test data number {count_index} out of {len(datatest)}')
        dokum_file.write(f'Test data number {count_index} index of {dataset_index}\n')
        k_input = []
        rand_result.append((random.choice(labels),dataset_row["label"]))
        for k in k_ways:

            question = "".join(dataset_row["query"].split(":")[:-1]).strip() + " : "
            
            k_input.append(question)
            
            print(f'K current value {k} length of input is {len(question.split())}')
            dokum_file.write(f'K current value {k} length of input is {len(question.split())}\n')

        answer_map = {}
        exception_map = {}

        for k_index, inputs in enumerate(k_input):

            try:
              
              print(f"now k value of {k_value_hard_coded[k_index]}")
              # print("CUDA MEM RESERVED THIS PREDICTION!!!!")
              # print(torch.cuda.memory_reserved(0)/1e9)
              # k_past_context[k_value_hard_coded[k_index]]
              
              start = time.time()
              pred, normalized_scores = predict_class(model, tokenizer, [inputs], labels, None if k_value_hard_coded[k_index] != 0 else None, k_past_values[k_value_hard_coded[k_index]] if k_value_hard_coded[k_index] != 0 else None, k_value_hard_coded[k_index], k_context_length[k_value_hard_coded[k_index]], cuda=cuda)  
              end = time.time()
              print(f"PREDICTION TAKES {end - start} seconds")
              
              if k_value_hard_coded[k_index] == 0 :
                print("===INPUT===")
                print(inputs)
                print("===TRUE===")
                print(dataset_row["label"])
                dokum_file.write("===INPUT===\n")
                dokum_file.write(inputs + "\n")
                dokum_file.write("===TRUE===\n")
                dokum_file.write(dataset_row["label"] + "\n")
              print(f"===PRED K={k_value_hard_coded[k_index]}===")
              print(pred)
              print(normalized_scores)
              dokum_file.write("===PRED===\n")
              dokum_file.write(pred + "\n")
              dokum_file.write(str(normalized_scores) + "\n")

              answer_map[k_value_hard_coded[k_index]] = [dataset_row["label"] , pred]

              exception_map[k_value_hard_coded[k_index]] = 0
              
            except KeyboardInterrupt:

              raise ValueError("Keyboard Interrupt")

            except Exception:

              answer_map[k_value_hard_coded[k_index]] = ["invalid" , "invalid"]

              exception_map[k_value_hard_coded[k_index]] = 1
       
        analysis[dataset_index] = copy.copy(answer_map)
        exceptions[dataset_index] = copy.copy(exception_map)
    
    dokum_file.close()
    
    return analysis, exceptions, rand_result

import time
def evaluate_task(task_name, task_dataset, task_dataset_eval, task_labels, configs, path_save, compare_indobert=False, DatasetClass=None, DataLoaderClass=None, delimiter='\n'):
  start = time.time()
  analysis_task = []
  exceptions_task = []
  random_task = []
  k_value_hard_coded = configs["k_value_hard_coded"]

  for repeat in range(configs.get("normalize_run_repeat", 5)):
    
        current_analysis_task, current_exceptions_task, current_random_task = evaluate_next_word_k_way(tokenizer, 
                                          k_value_hard_coded=configs["k_value_hard_coded"], 
                                          labels=task_labels, 
                                          predict_class=configs["predict_class"], 
                                          train_df=task_dataset,
                                          eval_df=task_dataset_eval, 
                                          dokum_file=path_save,
                                          n_test_case=configs["n_test_case"],
                                          compare_indobert=compare_indobert,
                                          DatasetClass=DatasetClass,
                                          DataLoaderClass=DataLoaderClass, 
                                          delimiter='\n')
        
        print(f"*********************NOW RUN NUMBER {repeat} ***********************")
        
        analysis_task.append(current_analysis_task)
        exceptions_task.append(current_exceptions_task)
        random_task.append(current_random_task)

  import numpy as np
  from sklearn.metrics import classification_report
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt

  # GET ConfusionMatrixDisplay FOR ALL THE RUN 

  for ind_run, data in enumerate(analysis_task):
    key   = f"{task_name} {ind_run}rd run"
    data  = analysis_task[0]

    print(f"Analysis for {key}")
    analysis_per_k = {}

    for case_index, case_value in data.items():
        for k_value, result in case_value.items():
            if k_value in analysis_per_k:
                analysis_per_k[k_value].append(result)
            else:
                analysis_per_k[k_value] = [result]

    # for pairs in analysis_per_k[1]:
    #   print(pairs)
                
    for k_value in analysis_per_k:
        
        print(f'FOR K VALUE = {k_value}')
        
        y_true = np.array(analysis_per_k[k_value]).T[0]
        y_pred = np.array(analysis_per_k[k_value]).T[1]

        print(y_pred)

        postprocess_y_true = []
        postprocess_y_pred = []

        for index, _ in enumerate(y_true):
          if y_true[index] != "invalid":
            postprocess_y_true.append(y_true[index])
            postprocess_y_pred.append(y_pred[index])
        
        print(postprocess_y_pred)

        target_names = task_labels
        
        print(classification_report(postprocess_y_true, postprocess_y_pred, target_names=target_names))
        confusion_matrix(postprocess_y_true, postprocess_y_pred)
        
        cm = confusion_matrix(postprocess_y_true, postprocess_y_pred, labels=target_names)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=target_names)
        
        disp.plot(cmap=plt.cm.Blues)
        plt.title(key + f'k_{k_value}')
        plt.savefig(path_save / (key + f'k_{k_value}.png'))
        plt.show()
  
  run_index2correct_k = {}
  run_index2n_test_results = {}

  # GET PLOT FOR EACH K VALUE AVERAGED EACH RUN 
  for run_index, current_run in enumerate(analysis_task):
    
    current_run_correct_k         = {}
    current_run_k_n_test_results  = {}
    
    for test_index, test_results in current_run.items():
      
        for k_index, pairs in test_results.items():
          if k_index not in current_run_correct_k:
            current_run_correct_k[k_index] = 0
            current_run_k_n_test_results[k_index] = 0
          
          current_run_k_n_test_results[k_index] += 1
          
          if pairs[0] == "invalid":
            current_run_k_n_test_results[k_index] -= 1
          elif pairs[0] == pairs[1]:
            current_run_correct_k[k_index] += 1
      
    run_index2correct_k[run_index] = current_run_correct_k
    run_index2n_test_results[run_index] = current_run_k_n_test_results

  # importing package
  import matplotlib.pyplot as plt
    
  for run_index, correct_k in run_index2correct_k.items():
    
    # create data
    x = [x+1 for x in list(correct_k.keys())]
    y = [x/(max(1, run_index2n_test_results[run_index][k_index]))*100 for k_index, x in list(correct_k.items())] 

    # plot line
    plt.plot(x, y)
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title(f"{task_name} run number " + str(run_index))
    plt.savefig(path_save / (f"{task_name}_run_number_" + str(run_index)))
    plt.show()

  total_correct_k = {}
  total_n_per_k = {}
  for run_index in run_index2correct_k:
    for k_val, correct_k in run_index2correct_k[run_index].items():
      if k_val not in total_correct_k:
        total_correct_k[k_val] = 0
        total_n_per_k[k_val] = 0
      total_correct_k[k_val] += correct_k
      total_n_per_k[k_val] += run_index2n_test_results[run_index][k_val]

  # create data
  x = [x+1 for x in list(total_correct_k.keys())]
  y = [x/total_n_per_k[k_index]*100 for k_index, x in list(total_correct_k.items())] 

  # plot line
  plt.plot(x, y)
  plt.xlabel("K Value")
  plt.ylabel("Accuracy")
  plt.title(f"{task_name} overall")
  plt.savefig(path_save / (f"{task_name}_averaged"))
  plt.show()

  # COUNT EXCEPTIONS:
  try:
    for index, run in enumerate(exceptions_task):
      exc_this_run = 0
      for exc_map in exceptions_task[run]:
        print(exc_map)
        for test_run in exceptions_task[exc_map]:
          exc_this_run+=1
      print(f"NUMBER OF EXCEPTION IN RUN {index}" + str(exc_this_run))
  except:
    pass

  end = time.time()
  dokum_file = open(path_save / "log_time.txt", "w")

  total_pred_data_point = configs["n_test_case"] * configs["normalize_run_repeat"]

  print(f"RUNTIME TOTAL {end - start} seconds, for {total_pred_data_point} test data point")
  dokum_file.write(f"RUNTIME TOTAL {end - start} seconds, for {total_pred_data_point} test data point")
  dokum_file.close()

# TEST ALL
import copy

indoNLU_datasets = {}
indoNLU_test_datasets = {}
indoNLU_labels = {}
indoNLU_data_classes = {}

configs = {
    "k_value_hard_coded" : [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 80], 
    # "predict_class" :  normalize,
    "predict_class" :  calculate_log_prob_global, 
    "n_test_case" : 10,
    "normalize_run_repeat" : 3
}



indoNLU_labels["emot_dataset"] = ['senang','marah','takut','sedih','cinta']
indoNLU_labels["emot_dataset_verbalized"] = ['senang','marah','takut','sedih','cinta']
indoNLU_labels["smsa_dataset"] = ["positif", "negatif", "netral"]
indoNLU_labels["smsa_dataset_verbalized"] = ["positif", "negatif", "netral"]
indoNLU_labels["wrete_dataset"] = ["benar", "salah"]
# indoNLU_labels["hoasa_dataset"] = ['positif','netral','ambigu','negatif']
# indoNLU_labels["hoasa_dataset_verbalized"] = ['positif','netral','ambigu','negatif']
# indoNLU_labels["casa_dataset"] = ['positif','netral','negatif']
# indoNLU_labels["casa_dataset_verbalized"] = ['positif','netral','negatif']

indoNLU_datasets["emot_dataset"] = create_emot(is_verbalized=False)
indoNLU_datasets["emot_dataset_verbalized"] = create_emot(is_verbalized=True)
indoNLU_datasets["smsa_dataset"] = create_smsa(is_verbalized=False)
indoNLU_datasets["smsa_dataset_verbalized"] = create_smsa(is_verbalized=True)
indoNLU_datasets["wrete_dataset"] = create_wrete()
# indoNLU_datasets["hoasa_dataset"] = create_hoasa(is_verbalized=False)
# indoNLU_datasets["hoasa_dataset_verbalized"] = create_hoasa(is_verbalized=True)
# indoNLU_datasets["casa_dataset"] = create_casa(is_verbalized=False)
# indoNLU_datasets["casa_dataset_verbalized"] = create_casa(is_verbalized=True)

indoNLU_datasets["emot_dataset"]["label"] = indoNLU_datasets["emot_dataset"].apply(lambda row : return_label_emot(row), axis=1)
indoNLU_datasets["emot_dataset_verbalized"]["label"] = indoNLU_datasets["emot_dataset_verbalized"].apply(lambda row : return_label_emot(row), axis=1)
indoNLU_datasets["smsa_dataset"]["label"] = indoNLU_datasets["smsa_dataset"].apply(lambda row : return_label_smsa(row), axis=1)
indoNLU_datasets["smsa_dataset_verbalized"]["label"] = indoNLU_datasets["smsa_dataset_verbalized"].apply(lambda row : return_label_smsa(row), axis=1)
indoNLU_datasets["wrete_dataset"]["label"] = indoNLU_datasets["wrete_dataset"].apply(lambda row : return_label_wrete(row), axis=1)
# indoNLU_datasets["hoasa_dataset"]["label"] = indoNLU_datasets["hoasa_dataset"].apply(lambda row : return_label_emot(row), axis=1)
# indoNLU_datasets["hoasa_dataset_verbalized"]["label"] = indoNLU_datasets["hoasa_dataset_verbalized"].apply(lambda row : return_label_emot(row), axis=1)
# indoNLU_datasets["casa_dataset"]["label"] = indoNLU_datasets["casa_dataset"].apply(lambda row : return_label_smsa(row), axis=1)
# indoNLU_datasets["casa_dataset_verbalized"]["label"] = indoNLU_datasets["casa_dataset_verbalized"].apply(lambda row : return_label_smsa(row), axis=1)

indoNLU_test_datasets["emot_dataset"] = create_emot_test(is_verbalized=False)
indoNLU_test_datasets["emot_dataset_verbalized"] = create_emot_test(is_verbalized=True)
indoNLU_test_datasets["smsa_dataset"] = create_smsa_test(is_verbalized=False)
indoNLU_test_datasets["smsa_dataset_verbalized"] = create_smsa_test(is_verbalized=True)
indoNLU_test_datasets["wrete_dataset"] = create_wrete_test() #create_wrete_test()
# indoNLU_test_datasets["hoasa_dataset"] = create_hoasa(is_verbalized=False)
# indoNLU_test_datasets["hoasa_dataset_verbalized"] = create_hoasa(is_verbalized=True)
# indoNLU_test_datasets["casa_dataset"] = create_casa(is_verbalized=False)
# indoNLU_test_datasets["casa_dataset_verbalized"] = create_casa(is_verbalized=True)

indoNLU_test_datasets["emot_dataset"]["label"] = indoNLU_test_datasets["emot_dataset"].apply(lambda row : return_label_emot(row), axis=1)
indoNLU_test_datasets["emot_dataset_verbalized"]["label"] = indoNLU_test_datasets["emot_dataset_verbalized"].apply(lambda row : return_label_emot(row), axis=1)
indoNLU_test_datasets["smsa_dataset"]["label"] = indoNLU_test_datasets["smsa_dataset"].apply(lambda row : return_label_smsa(row), axis=1)
indoNLU_test_datasets["smsa_dataset_verbalized"]["label"] = indoNLU_test_datasets["smsa_dataset_verbalized"].apply(lambda row : return_label_smsa(row), axis=1)
indoNLU_test_datasets["wrete_dataset"]["label"] = indoNLU_test_datasets["wrete_dataset"].apply(lambda row : return_label_wrete(row), axis=1)
# indoNLU_test_datasets["hoasa_dataset"]["label"] = indoNLU_test_datasets["hoasa_dataset"].apply(lambda row : return_label_emot(row), axis=1)
# indoNLU_test_datasets["hoasa_dataset_verbalized"]["label"] = indoNLU_test_datasets["hoasa_dataset_verbalized"].apply(lambda row : return_label_emot(row), axis=1)
# indoNLU_test_datasets["casa_dataset"]["label"] = indoNLU_test_datasets["casa_dataset"].apply(lambda row : return_label_smsa(row), axis=1)
# indoNLU_test_datasets["casa_dataset_verbalized"]["label"] = indoNLU_test_datasets["casa_dataset_verbalized"].apply(lambda row : return_label_smsa(row), axis=1)

indoNLU_data_classes["emot_dataset"] = (None, None) 
indoNLU_data_classes["emot_dataset_verbalized"] = (None, None) 
indoNLU_data_classes["smsa_dataset"] = (None, None) 
indoNLU_data_classes["smsa_dataset_verbalized"] = (None, None) 
indoNLU_data_classes["wrete_dataset"] = (None, None)  
# indoNLU_data_classes["hoasa_dataset"] = (DocumentSentimentDataset, DocumentSentimentDataLoader) 
# indoNLU_data_classes["hoasa_dataset_verbalized"] = (None, None)

for task in indoNLU_datasets:

  configs["n_test_case"] = int(min(configs["n_test_case"], indoNLU_test_datasets[task]["label"].value_counts().min()))

  path_save = save_configs(copy.copy(configs), model_name, task)

  evaluate_task(task, indoNLU_datasets[task], indoNLU_test_datasets[task], indoNLU_labels[task], configs, path_save, compare_indobert=False, DatasetClass=indoNLU_data_classes[task][0], DataLoaderClass=indoNLU_data_classes[task][1])
    