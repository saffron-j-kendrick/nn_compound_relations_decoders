import data_utils
import os
import numpy as np
import torch
import tqdm
import pathlib
import argparse
import inflect
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import model_utils
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for extracting representations')
parser.add_argument('--models', nargs='+', type=str, default=None, help='Models to use ("roberta-base", "xlnet-base-cased", and/or "xlm-mlm-xnli15-1024")')
parser.add_argument('--layers', nargs='+', type=int, default=None, help='Which layers to use. Defaults to all excluding embedding. First layer indexed by 1')
parser.add_argument('--device', type=str, default="cpu", help='"cuda" or "cpu"')
parser.add_argument('--representations', nargs='+', type=str, default=None, help='Representations to extract ("cls", "mean_pooled", "noun_noun_compounds", "compositional_analysis)')
parser.add_argument('--rep_loc', type=str, default="./data", help='Where to save representations')
parser.add_argument('--save_attention', dest='save_attention', action='store_true', default=False)
parser.add_argument('--load_if_available', dest='load_if_available', action='store_true', default=False)
parser.add_argument('--amount_of_dataset', type=float, default=1.0, help='Proportion of dataset to use')



def get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, load_if_available=True, batch_size = 1, rep_loc='./data', rep_type="mean_pooled", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=mean_pool_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, save_attention=save_attention)

def get_final_mod_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="final_mod", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_mod_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, save_attention=save_attention)

def get_final_head_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="final_head", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_head_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, save_attention=save_attention)

def get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="noun_noun_compound", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=noun_noun_compound_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, middle_dim=2, save_attention=save_attention)

def get_noun_noun_compound_final_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="noun_noun_final_compound", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=noun_noun_final_compound_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, middle_dim=2, save_attention=save_attention)


def get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers, load_if_available=True, batch_size = 1, rep_type="final_word", rep_loc='./data', torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_word_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence_and": corrected_form_compounds_per_sentence_and}, save_attention=save_attention)




def get_noun_noun_compound_and_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, head_and_words_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="noun_noun_and_compound", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=noun_noun_and_compound_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence_and": head_and_words_per_sentence}, middle_dim=2, save_attention=save_attention)


def get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, word_per_sentence, layers, load_if_available=True, batch_size = 1, rep_type="compositional_analysis_and", rep_loc='./data', torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=single_word_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"word_per_sentence": word_per_sentence}, save_attention=save_attention)


def get_correct_form(word, sentence, stemmer):

    try:
        correct_form = word_tokenize(sentence)[np.where(np.array([stemmer.stem(x) for x in word_tokenize(sentence)]) == stemmer.stem(word))[0][0]]
    except IndexError:
        return ''

    return correct_form

def get_corrected_form_compounds_per_sentence(sentences, mod_head_words_per_sentence):
    # 
    stemmer = PorterStemmer()

    # Get first matching stem
    correct_mod_words = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence)]
    correct_head_nouns = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence)]

    res = list(zip(sentences, mod_head_words_per_sentence[:, 0], correct_mod_words, mod_head_words_per_sentence[:, 1], correct_head_nouns))
    
def search_sequence_numpy(arr,seq):
    # https://stackoverflow.com/a/36535397
    # 
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []   



def get_corrected_form_compounds_and_per_sentence(sentences, head_and_words_per_sentence):
    # 
    stemmer = PorterStemmer()

    # Get first matching stem
    correct_and_words = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(head_and_words_per_sentence)]
    correct_head_nouns = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(head_and_words_per_sentence)]

    res = list(zip(sentences, head_and_words_per_sentence[:, 0], correct_head_nouns, head_and_words_per_sentence[:, 1], correct_and_words))


def single_word_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, word_per_sentence):
    # Get tokens where tokens aren't special tokens or pad tokens

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]

    words_per_batch = np.array(word_per_sentence)[i:i+batch_size]
    word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(words_per_batch.tolist())['input_ids'])
    
    # Remove special tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in word_input_ids_per_sent_raw]

    word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(word_input_ids_per_sent)]

    word_reps = np.vstack([np.mean(reps[word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
    
    return word_reps

def noun_noun_compound_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
    # Get tokens where tokens aren't special tokens or pad tokens

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    
    # Remove special tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]

    # if model_name == 'xlm-mlm-xnli15-1024':
    #     head_noun_input_ids_per_sent = [x[1:] for x in head_noun_input_ids_per_sent]
    #     mod_word_input_ids_per_sent = [x[1:] for x in mod_word_input_ids_per_sent]

    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
    mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]

    head_noun_reps = np.vstack([np.mean(reps[head_noun_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
    mod_word_reps = np.vstack([np.mean(reps[mod_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
    compound_reps = np.stack([mod_word_reps, head_noun_reps], axis=1)
    
    # Shape = (batch_size, 2, hidden_size)
    return compound_reps


def noun_noun_and_compound_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence_and):

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]  
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence_and])[i:i+batch_size]

    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    and_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    and_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in and_word_input_ids_per_sent_raw]
    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
    and_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(and_word_input_ids_per_sent)]
    head_noun_reps = np.vstack([reps[head_noun_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    and_word_reps = np.vstack([np.mean(reps[and_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])

    compound_and_reps = np.stack([head_noun_reps, and_word_reps], axis=1)
    return compound_and_reps

def noun_noun_final_compound_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
    # Get tokens where tokens aren't special tokens or pad tokens

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the modifier word tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]
    mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]
    
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]

    mod_word_token_counts = [len(x) for x in mod_word_input_ids_per_sent]
    head_noun_token_counts = [len(x) for x in head_noun_input_ids_per_sent]

    #print(mod_word_token_counts)


    # If there is only one head noun token, return the token, if there is more than one head noun token, return the last token
    head_noun_reps = np.vstack([reps[head_noun_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    # If there is only one modifier token, return the token, if there is more than one modifier token, return the last token
    mod_word_reps = np.vstack([reps[mod_word_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    final_compound_reps = np.stack([mod_word_reps, head_noun_reps], axis=1)
    return mod_word_token_counts, head_noun_token_counts




def mean_pool_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i=None):
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    
    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    return np.vstack([np.mean(reps[get_tokens_to_keep(input_ids[i])].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])

def final_mod_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
    
    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the modifier word tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]
    mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]

    # If there is only one modifier token, return the token, if there is more than one modifier token, return the last token
    mod_word_reps = np.vstack([reps[mod_word_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return mod_word_reps

def final_head_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the head noun tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]

    # If there is only one head noun token, return the token, if there is more than one head noun token, return the last token
    head_noun_reps = np.vstack([reps[head_noun_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return head_noun_reps

def final_word_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence_and):

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence_and])[i:i+batch_size]
    

    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)

    final_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    final_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in final_word_input_ids_per_sent_raw]
    final_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(final_word_input_ids_per_sent)]
    final_word_reps = np.vstack([reps[final_word_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return final_word_reps






def get_tokens_from_layers(
    model_name, model, tokeniser, input_ids, attention_mask, layers,
    token_selector=mean_pool_selector, load_if_available=True, batch_size=1,
    rep_type="sentence_pair_cls", torch_device="cuda", save_reps=True,
    data_loc='./data', add_arg_dict={}, middle_dim=None, save_attention=False
):
    """
    Extracts token representations from specified layers of a model and optionally saves them.
    """
    # Get file locations for layer representations
    rep_locs_per_layer = [
        data_utils.get_hidden_state_file(model_name, layer=x, rep_type=rep_type, data_loc=data_loc)
        for x in layers
    ]
    load_reps = load_if_available and all(os.path.isfile(x) for x in rep_locs_per_layer)

    if load_reps:
        return [np.load(x) for x in rep_locs_per_layer]

    print(f'Extracting representations from model for layers {layers}')
    
    # Move data to the appropriate device
    input_ids = input_ids.to(torch_device)
    attention_mask = attention_mask.to(torch_device)
    model.to(torch_device)

    # Initialize token representations dynamically
    tokens_per_layer = [
        np.zeros((input_ids.shape[0], model.config.hidden_size))
        for _ in layers
    ]
    
    # Optionally initialize attention representations
    if save_attention:
        seq_len = max(len(tokeniser.decode(x)) for x in input_ids)
        attention_per_layer = [-np.ones((input_ids.shape[0], seq_len**2)) for _ in layers]

    # Extract representations
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
            outputs = model(
                input_ids[i:i+batch_size],
                attention_mask=attention_mask[i:i+batch_size],
                output_attentions=save_attention,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[1:]  # Exclude embeddings
            add_arg_dict["i"] = i

            # Process each layer
            for idx, layer in enumerate(layers):
                tokens_per_layer[idx][i:i+batch_size] = token_selector(
                    model, model_name, tokeniser, hidden_states,
                    input_ids[i:i+batch_size], layer, batch_size, **add_arg_dict
                )

    # Save representations
    if save_reps:
        for idx, layer in enumerate(layers):
            layer_file = data_utils.get_hidden_state_file(
                model_name, layer=layer, rep_type=rep_type, data_loc=data_loc
            )
            pathlib.Path('/'.join(layer_file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            np.save(layer_file, tokens_per_layer[idx])

    if save_attention:
        for idx, layer in enumerate(layers):
            atten_file = data_utils.get_hidden_state_file(
                model_name, layer=layer, rep_type=f"{rep_type}_attention", data_loc=data_loc
            )
            pathlib.Path('/'.join(atten_file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            np.save(atten_file, attention_per_layer[idx])

    return tokens_per_layer





def extract_and_save_representations(amount_of_dataset=1, batch_size=1, models=None, load_if_available=False, layers=None, torch_device="cpu", representations=["mean_pooled"], rep_loc='./data', save_attention=False):

    if representations == None:
        representations = ["final_mod_tokens", "final_head_tokens"]

   
    sentences = data_utils.get_noun_noun_that_compounds_sentences()
    mod_head_words_per_sentence = data_utils.get_noun_noun_mod_head_words_per_sentence()
    head_and_words_per_sentence = data_utils.get_noun_noun_head_that_words_per_sentence()
    num_to_keep = int(amount_of_dataset * len(sentences))
    sentences = sentences[:num_to_keep]
    mod_head_words_per_sentence = mod_head_words_per_sentence[:num_to_keep]
    head_and_words_per_sentence = head_and_words_per_sentence[:num_to_keep]

    if "compositional_analysis" in representations:
       
        words_to_find, comp_sentences = data_utils.get_compositional_probe_words_and_sentences()
        num_to_keep = int(amount_of_dataset * len(comp_sentences))
        words_to_find = words_to_find[:num_to_keep]
        comp_sentences = comp_sentences[:num_to_keep]

    elif "compositional_analysis_and" in representations:
       
        words_to_find, comp_sentences = data_utils.get_compositional_probe_words_and_sentences_and()
        num_to_keep = int(amount_of_dataset * len(comp_sentences))
        words_to_find = words_to_find[:num_to_keep]
        comp_sentences = comp_sentences[:num_to_keep]

    initial_layers = layers

    if models == None:
        models = model_utils.dev_model_configs.keys()

    for model_name in tqdm.tqdm(models):
        print('Loading {}'.format(model_name))
        model, tokeniser = model_utils.load_model(model_name)
        if tokeniser.pad_token is None:
            if tokeniser.eos_token:
                tokeniser.pad_token = tokeniser.eos_token
            else:
                tokeniser.add_special_tokens({'pad_token': '<pad>'})

        #unpack_dict = lambda x: (x['input_ids'], x['attention_mask'])

        if initial_layers == None:
            layers = range(1, model.config.num_hidden_layers + 1)

        layers = [x for x in layers if x in range(1, model.config.num_hidden_layers + 1)]
        
        if "mean_pooled" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, save_attention=save_attention, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds_mean_pooled', rep_loc=rep_loc, torch_device=torch_device)
        
        if "final_mod_tokens" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_corrected_form_compounds_per_sentence()
            get_final_mod_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_mod', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        
        if "final_head_tokens" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_corrected_form_compounds_per_sentence()
            get_final_head_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_head', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        if "final_word_tokens" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence_and = data_utils.load_corrected_form_compounds_per_sentence_that()
            get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_word', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        # if "final_word_tokens" in representations:
        #     inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
        #     input_ids = inputs["input_ids"]
        #     attention_mask = inputs["attention_mask"]
        #     #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
        #     corrected_form_compounds_per_sentence_and = data_utils.load_corrected_form_compounds_per_sentence_and()
        #     get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_word', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        # if "final_but_tokens" in representations:
        #     inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
        #     input_ids = inputs["input_ids"]
        #     attention_mask = inputs["attention_mask"]
        #     #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
        #     corrected_form_compounds_per_sentence_and = data_utils.load_corrected_form_compounds_per_sentence_but()
        #     get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_word', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        # if "final_that_tokens" in representations:
        #     inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
        #     input_ids = inputs["input_ids"]
        #     attention_mask = inputs["attention_mask"]
        #     #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
        #     corrected_form_compounds_per_sentence_and = data_utils.load_corrected_form_compounds_per_sentence_that()
        #     get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_word', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)



        if "noun_noun_compounds" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_corrected_form_compounds_per_sentence()
            get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        if "noun_noun_final_compounds" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_corrected_form_compounds_per_sentence()
            get_noun_noun_compound_final_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_final_compounds', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        if "noun_noun_and_compounds" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence_and = data_utils.load_corrected_form_compounds_per_sentence_and()
            get_noun_noun_compound_and_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_and_compounds', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        
        if "compositional_analysis" in representations:
            inputs = tokeniser(comp_sentences,  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(comp_sentences, max_length=512, return_tensors='pt', pad_to_max_length=True))
            get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, words_to_find, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compositional_analysis', torch_device=torch_device, save_attention=save_attention, rep_loc=rep_loc)
        
        if "compositional_analysis_and" in representations:
            inputs = tokeniser(comp_sentences,  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(comp_sentences, max_length=512, return_tensors='pt', pad_to_max_length=True))
            get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, words_to_find, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compositional_analysis_and', torch_device=torch_device, save_attention=save_attention, rep_loc=rep_loc)
        


        else:
            print('No representations specified')

if __name__ == "__main__":
    args = parser.parse_args()
    print(vars(args))

    extract_and_save_representations(amount_of_dataset=args.amount_of_dataset, load_if_available=args.load_if_available, batch_size=args.batch_size, models=args.models, layers=args.layers, torch_device=args.device, representations=args.representations, rep_loc=args.rep_loc, save_attention=args.save_attention)