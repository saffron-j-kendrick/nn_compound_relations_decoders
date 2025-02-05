# import data_utils
# import os
# import numpy as np
# import torch
# import tqdm
# import pathlib
# import argparse
# import inflect
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize

# import model_utils
# import data_utils

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=1, help='Batch size for extracting representations')
# parser.add_argument('--models', nargs='+', type=str, default=None, help='Models to use ("roberta-base", "xlnet-base-cased", and/or "xlm-mlm-xnli15-1024")')
# parser.add_argument('--layers', nargs='+', type=int, default=None, help='Which layers to use. Defaults to all excluding embedding. First layer indexed by 1')
# parser.add_argument('--device', type=str, default="cpu", help='"cuda" or "cpu"')
# parser.add_argument('--representations', nargs='+', type=str, default=None, help='Representations to extract ("cls", "mean_pooled", "noun_noun_compounds", "compositional_analysis)')
# parser.add_argument('--rep_loc', type=str, default="./data", help='Where to save representations')
# parser.add_argument('--save_attention', dest='save_attention', action='store_true', default=False)
# parser.add_argument('--load_if_available', dest='load_if_available', action='store_true', default=False)
# parser.add_argument('--amount_of_dataset', type=float, default=1.0, help='Proportion of dataset to use')

# def get_cls_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, load_if_available=True, batch_size = 1, rep_loc='./data', rep_type="sentence_pair_cls", torch_device="cuda"):
#     return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
#      token_selector=cls_token_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device)

# def get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, load_if_available=True, batch_size = 1, rep_loc='./data', rep_type="mean_pooled", torch_device="cuda", save_attention=False):
#     return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
#      token_selector=mean_pool_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, save_attention=save_attention)
     
# def get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="noun_noun_compound", torch_device="cuda", save_attention=False):
#     return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
#      token_selector=noun_noun_compound_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
#      add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, middle_dim=2, save_attention=save_attention)

# def get_compound_and_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="compound_and", torch_device="cuda", save_attention=False):
#     return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
#      token_selector=compound_and_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
#      add_arg_dict={"corrected_form_compounds_per_sentence_and": corrected_form_compounds_per_sentence_and}, middle_dim=2, save_attention=save_attention)



# def get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, word_per_sentence, layers, load_if_available=True, batch_size = 1, rep_type="compositional_analysis_and", rep_loc='./data', torch_device="cuda", save_attention=False):
#     return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
#      token_selector=single_word_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
#      add_arg_dict={"word_per_sentence": word_per_sentence}, save_attention=save_attention)

# def get_correct_form(word, sentence, stemmer):

#     try:
#         correct_form = word_tokenize(sentence)[np.where(np.array([stemmer.stem(x) for x in word_tokenize(sentence)]) == stemmer.stem(word))[0][0]]
#     except IndexError:
#         return ''

#     return correct_form

# def get_corrected_form_compounds_per_sentence(sentences, mod_head_words_per_sentence):
#     # 
#     stemmer = PorterStemmer()

#     # Get first matching stem
#     correct_mod_words = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence)]
#     correct_head_nouns = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence)]

#     res = list(zip(sentences, mod_head_words_per_sentence[:, 0], correct_mod_words, mod_head_words_per_sentence[:, 1], correct_head_nouns))

# def get_corrected_form_compounds_and_per_sentence(sentences, head_and_words_per_sentence):
#     # 
#     stemmer = PorterStemmer()

#     # Get first matching stem
#     correct_and_words = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(head_and_words_per_sentence)]
#     correct_head_nouns = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(head_and_words_per_sentence)]

#     res = list(zip(sentences, head_and_words_per_sentence[:, 0], correct_head_nouns, head_and_words_per_sentence[:, 1], correct_and_words))


# def search_sequence_numpy(arr,seq):
#     # https://stackoverflow.com/a/36535397
#     # 
#     # Store sizes of input array and sequence
#     Na, Nseq = arr.size, seq.size

#     # Range of sequence
#     r_seq = np.arange(Nseq)

#     # Create a 2D array of sliding indices across the entire length of input array.
#     # Match up with the input sequence & get the matching starting indices.
#     M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

#     # Get the range of those indices as final output
#     if M.any() > 0:
#         return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
#     else:
#         return []   

# def single_word_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, word_per_sentence):
#     # Get tokens where tokens aren't special tokens or pad tokens

#     if model_name in ['distilroberta-base', 'xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/gpt2"]:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]

#     words_per_batch = np.array(word_per_sentence)[i:i+batch_size]
#     word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(words_per_batch.tolist())['input_ids'])
    
#     # Remove special tokens
#     non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#     get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
#     word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in word_input_ids_per_sent_raw]

#     # if 'xlm' in model_name:
#     #     word_input_ids_per_sent = [x[1:] for x in word_input_ids_per_sent]

#     word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(word_input_ids_per_sent)]

#     word_reps = np.vstack([np.mean(reps[word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])

#     return word_reps

# def noun_noun_compound_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
#     # Get tokens where tokens aren't special tokens or pad tokens

#     if model_name in ['distilroberta-base','xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/gpt2", "microsoft/phi-1"]:
#         layer_reps = token_reps[layer].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    
#     compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    
#     head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
#     mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    
#     # Remove special tokens
#     non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#     get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
#     head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
#     mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]
     
#     head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
#     mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]

#     head_noun_reps = np.vstack([np.mean(reps[head_noun_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     mod_word_reps = np.vstack([np.mean(reps[mod_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     compound_reps = np.stack([mod_word_reps, head_noun_reps], axis=1)

#     # Shape = (batch_size, 2, hidden_size)
#     return compound_reps


# def compound_and_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence_and):
#     # Get tokens where tokens aren't special tokens or pad tokens

#     if model_name in ['distilroberta-base','xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/gpt2", "microsoft/phi-1"]:
#         layer_reps = token_reps[layer].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    
#     compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence_and])[i:i+batch_size]
    
#     head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
#     and_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    
#     # Remove special tokens
#     non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#     get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
#     head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
#     and_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in and_word_input_ids_per_sent_raw]
     
#     head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
#     and_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(and_word_input_ids_per_sent)]

#     head_noun_reps = np.vstack([np.mean(reps[head_noun_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     and_word_reps = np.vstack([np.mean(reps[and_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     compound_and_reps = np.stack([head_noun_reps, and_word_reps], axis=1)

#     # Shape = (batch_size, 2, hidden_size)
#     # print("compound reps shape", compound_and_reps.shape)
#     return compound_and_reps



# def compound_but_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence_but):
#     # Get tokens where tokens aren't special tokens or pad tokens

#     if model_name in ['distilroberta-base','xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/gpt2"]:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    
#     compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence_but])[i:i+batch_size]
    
#     head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
#     but_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    
#     # Remove special tokens
#     non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#     get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
#     head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
#     but_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in but_word_input_ids_per_sent_raw]
     
#     head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
#     but_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(but_word_input_ids_per_sent)]

#     head_noun_reps = np.vstack([np.mean(reps[head_noun_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     but_word_reps = np.vstack([np.mean(reps[but_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     compound_but_reps = np.stack([head_noun_reps, but_word_reps], axis=1)

#     # Shape = (batch_size, 2, hidden_size)
#     # print("compound reps shape", compound_and_reps.shape)
#     return compound_but_reps


# def compound_that_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence_that):
#     # Get tokens where tokens aren't special tokens or pad tokens

#     if model_name in ['distilroberta-base','xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/gpt2"]:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    
#     compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence_that])[i:i+batch_size]
    
#     head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
#     that_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    
#     # Remove special tokens
#     non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#     get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
#     head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
#     that_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in that_word_input_ids_per_sent_raw]
     
#     head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
#     that_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(that_word_input_ids_per_sent)]

#     head_noun_reps = np.vstack([np.mean(reps[head_noun_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     that_word_reps = np.vstack([np.mean(reps[that_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
#     compound_but_reps = np.stack([head_noun_reps, that_word_reps], axis=1)

#     # Shape = (batch_size, 2, hidden_size)
#     # print("compound reps shape", compound_and_reps.shape)
#     return compound_but_reps



# def mean_pool_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i=None):
#     # Get tokens where tokens aren't special tokens or pad token
#     non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#     pad_token_mask = lambda x: np.array(x.cpu() == tokeniser.pad_token_id)
#     get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x) * (pad_token_mask(x) == False)).reshape(-1)

#     if model_name in ['distilroberta-base', 'xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/openai-gpt", "microsoft/phi-1", "openai-community/gpt2"]:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]

#     # For each sample in batch, select reps that don't correspond to special tokens or pad tokens and take the mean across the tokens
#     return np.vstack([np.mean(reps[get_tokens_to_keep(input_ids[i])].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])

# def cls_token_selector(model, model_name, tokeniser, token_reps, input_ids, layer, i, batch_size):
    
#     cls_token_locs = np.argwhere(input_ids.cpu().numpy()==tokeniser.cls_token_id)
    
#     first_cls_token_loc_per_sample = np.array([cls_token_locs[cls_token_locs[:, 0] == i][0] for i in range(batch_size)])
#     #print(token_reps.last_hidden_state.size())
#     if model_name in ['distilroberta-base', 'xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/openai-gpt"]:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]

#     #print(layer_reps.shape)

#     return np.vstack([sample[first_cls_token_loc_per_sample[i, -1]] for i, sample in enumerate(layer_reps)])


# def full_stop_token_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size):
#     # Find locations of the full stop token in the input IDs
#     full_stop_token_id = tokeniser.convert_tokens_to_ids(".")
#     full_stop_token_locs = np.argwhere(input_ids.cpu().numpy() == full_stop_token_id)
    
#     # Collect the first occurrence of the full stop token per sample in the batch
#     full_stop_token_loc_per_sample = [
#         full_stop_token_locs[full_stop_token_locs[:, 0] == i][0] if any(full_stop_token_locs[:, 0] == i) else None
#         for i in range(batch_size)
#     ]
    
#     # Handle model-specific layer representations
#     if model_name in ['distilroberta-base', 'xlm-mlm-xnli15-1024']:
#         layer_reps = token_reps[1][layer].cpu()[:, :, :]
#     elif layer == model.config.num_hidden_layers:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     elif model_name in ["meta-llama/Llama-3.2-1B", "openai-community/gpt2"]:
#         layer_reps = token_reps[0].cpu()[:, :, :]
#     else:
#         layer_reps = token_reps[2][layer].cpu()[:, :, :]

#     # # Extract the full stop token representation for each sample in the batch
#     # full_stop_reps = np.vstack([
#     #     sample[loc[-1]] if loc is not None else np.zeros((layer_reps.shape[-1],))
#     #     for loc, sample in zip(full_stop_token_loc_per_sample, layer_reps)
#     # ])

#     # Shape: (batch_size, hidden_size)
#     return np.vstack([
#         sample[loc[-1]] if loc is not None else np.zeros((layer_reps.shape[-1],))
#         for loc, sample in zip(full_stop_token_loc_per_sample, layer_reps)
#     ])

# def get_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, token_selector=cls_token_selector, load_if_available=True, batch_size = 1, rep_type="sentence_pair_cls", torch_device="cuda", save_reps=True, data_loc='./data', add_arg_dict={}, middle_dim=None, save_attention=False):
#     '''
#     layer 0: embedding
#     layer 1: first layer
#     . . . 
#     layer 12: usually the final layer
#     NOTE: Would probably be much faster to extract all layers at the same time (assuming we want outputs from all layers)
#     '''
#     rep_locs_per_layer = [data_utils.get_hidden_state_file(model_name, layer=x, rep_type=rep_type, data_loc=data_loc) for x in layers]
#     load_reps = load_if_available and os.path.isfile(rep_loc)

#     layer_map = dict(zip(layers, range(len(layers))))

#     if load_reps:
#         tokens_per_layer = [np.load(x) for x in rep_locs_per_layer]
#     else:
#         print('Extracting representations from model for layers {}'.format(layers))
        
#         input_ids = input_ids.to(torch_device)
#         attention_mask = attention_mask.to(torch_device)
#         model.to(torch_device)
        
#         tokens_per_layer = [np.zeros((input_ids.shape[0], model.config.hidden_size)) for x in layers]

#         # # # # TODO: Work this out from model object - for encoders
#         # embedding_size = 1024 if 'xlm' in model_name else 768
        
#         # tokens_per_layer = [np.zeros((input_ids.shape[0], embedding_size)) if middle_dim == None else np.zeros((input_ids.shape[0], middle_dim, embedding_size)) for x in layers]

#         if save_attention:
#             # Calculate size of attention matrix
#             non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
#             pad_token_mask = lambda x: np.array(x.cpu() == tokeniser.pad_token_id)
#             get_tokens_to_keep = lambda x: np.argwhere((pad_token_mask(x) == False)).reshape(-1)
#             tokens_to_keep_per_sample = [get_tokens_to_keep(x) for x in input_ids]
#             seq_lens = np.array([len(x) for x in tokens_to_keep_per_sample])
#             seq_len = max(seq_lens)
#             attention_per_layer = [-np.ones((input_ids.shape[0], seq_len**2)) for x in layers]

#             decode_tokens = lambda x: [tokeniser.decode(token) for token in x[get_tokens_to_keep(x)].tolist()]
#             '\n'.join([''.join(decode_tokens(x)) for x in input_ids])

#         with torch.no_grad():
#             for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
                                
#                 # this works for deecoder models, assuming save_attention is set to False
#                 output = model(input_ids[i:i+batch_size, :].reshape(batch_size, -1), attention_mask=attention_mask[i:i+batch_size, :].reshape(batch_size, -1), output_hidden_states=True)
                
#                 for layer in layers:
#                     # fetch the hidden states which are the output embeddings for the layer
#                     layer_reps = output.hidden_states[layer-1]
#                     tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = np.mean(layer_reps.cpu().numpy(), axis=1)

#                 # # this is for encoder models
#                 # token_reps = model(input_ids[i:i+batch_size, :].reshape(batch_size, -1), attention_mask=attention_mask[i:i+batch_size, :].reshape(batch_size, -1), output_attentions=save_attention)
                
#                 # add_arg_dict["i"] = i
#                 # for layer in layers:
#                 #     tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = token_selector(model, model_name, tokeniser, token_reps, input_ids[i:i+batch_size, :].reshape(batch_size, -1), layer, batch_size = batch_size, **add_arg_dict)
#                 # # if save_attention:
#                 # #     for layer in layers:
#                 # #         tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = token_selector(model, model_name, tokeniser, token_reps[:-1], input_ids[i:i+batch_size, :].reshape(batch_size, -1), layer, batch_size, **add_arg_dict)
#                 # #         for batch_offset, sample_attention in enumerate(token_reps[-1][layer - 1]):
#                 # #             sample_i = i + batch_offset
#                 # #             sample_len = seq_lens[sample_i]
#                 # #             attention_per_layer[layer_map[layer]][sample_i, :sample_len**2] = sample_attention.mean(axis=0)[tokens_to_keep_per_sample[sample_i]][:, tokens_to_keep_per_sample[sample_i]].reshape(sample_len**2).cpu()
#                 # # else:
#                 # #     for layer in layers:
#                 # #        tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = token_selector(model, model_name, tokeniser, token_reps, input_ids[i:i+batch_size, :].reshape(batch_size, -1), layer, batch_size = batch_size, **add_arg_dict)
        
#         if save_reps:
#             for layer in layers:
#                 pathlib.Path('/'.join(rep_locs_per_layer[layer_map[layer]].split('/')[:-1])).mkdir(parents=True, exist_ok=True)
#                 np.save(rep_locs_per_layer[layer_map[layer]], tokens_per_layer[layer_map[layer]])
    
#     if save_attention:
#         for layer in layers:
#             atten_rep_loc = data_utils.get_hidden_state_file(model_name, layer=layer, rep_type=rep_type + '_attention', data_loc=data_loc)
#             pathlib.Path('/'.join(atten_rep_loc.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
#             np.save(atten_rep_loc, attention_per_layer[layer_map[layer]])

#         return tokens_per_layer, attention_per_layer
#     else:
#         return tokens_per_layer

# def get_final_layer_cls_tokens(model_name, model, input_ids, attention_mask, load_if_available=True, 
#         batch_size = 1):

#     rep_loc = data_utils.get_hidden_state_file(model_name)
#     load_reps = load_if_available and os.path.isfile(rep_loc)

#     if load_reps:
#         cls_tokens = np.load(rep_loc)
#     else:
#         print('{} not found. Extracting reps.'.format(rep_loc))

#         # TODO: Work this out from model object
#         embedding_size = 40479 if 'gpt' in model_name else 768
#         cls_tokens = np.zeros((input_ids.shape[0], embedding_size))

#         with torch.no_grad():
#             for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
#                 token_reps = model(input_ids[i:i+batch_size, :].reshape(batch_size, -1), attention_mask=attention_mask[i:i+batch_size, :].reshape(batch_size, -1))
#                 cls_tokens[i:i+batch_size, :] = token_reps[0][:, 0, :]

#         pathlib.Path('/'.join(rep_loc.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
#         np.save(rep_loc, cls_tokens)
    
#     return cls_tokens

# def extract_and_save_representations(amount_of_dataset=1,batch_size=1, models=None, load_if_available=False, layers=None, torch_device="cpu", representations=["mean_pooled"], rep_loc='./data', save_attention=False):

#     if representations == None: 
#         representations = ["mean_pooled"]

#     sentences = data_utils.get_noun_noun_compound_sentences()
#     mod_head_words_per_sentence = data_utils.get_noun_noun_mod_head_words_per_sentence()
#     head_and_words_per_sentence = data_utils.get_noun_noun_head_and_words_per_sentence()
#     head_but_words_per_sentence = data_utils.get_noun_noun_head_but_words_per_sentence()
#     head_that_words_per_sentence = data_utils.get_noun_noun_head_that_words_per_sentence()

#     num_to_keep = int(amount_of_dataset * len(sentences))
#     sentences = sentences[:num_to_keep]
#     mod_head_words_per_sentence = mod_head_words_per_sentence[:num_to_keep]
#     head_and_words_per_sentence = head_and_words_per_sentence[:num_to_keep]
#     head_but_words_per_sentence = head_but_words_per_sentence[:num_to_keep]
#     head_that_words_per_sentence = head_that_words_per_sentence[:num_to_keep]

#     if "compositional_analysis_and" in representations:
#         words_to_find, comp_sentences = data_utils.get_compositional_probe_words_and_sentences_and()

#         num_to_keep = int(amount_of_dataset * len(comp_sentences))
#         words_to_find = words_to_find[:num_to_keep]
#         comp_sentences = comp_sentences[:num_to_keep]

#     initial_layers = layers

#     if models == None:
#         models = model_utils.dev_model_configs.keys()

#     for model_name in tqdm.tqdm(models):
#         print('Loading {}'.format(model_name))
#         model, tokeniser = model_utils.load_model(model_name)
#         if tokeniser.pad_token is None:
#             if tokeniser.eos_token:
#                 tokeniser.pad_token = tokeniser.eos_token
#             else:
#                 tokeniser.add_special_tokens({'pad_token': '<pad>'})
#         model.resize_token_embeddings(len(tokeniser))
#         unpack_dict = lambda x: (x['input_ids'], x['attention_mask'])

#         if initial_layers == None:
#             layers = range(1, model.config.num_hidden_layers + 1)

#         layers = [x for x in layers if x in range(1, model.config.num_hidden_layers + 1)]
        
#         if "cls" in representations:
#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
#             get_cls_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds_cls', rep_loc=rep_loc,torch_device=torch_device)
#         if "mean_pooled" in representations:
#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
#             get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, save_attention=save_attention, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds_mean_pooled', rep_loc=rep_loc, torch_device=torch_device)
#         if "noun_noun_compounds" in representations:
            
#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
#             corrected_form_compounds_per_sentence = data_utils.load_corrected_form_compounds_per_sentence()
#             get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)

#         if "compound_and" in representations:

#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors = 'pt', pad_to_max_length=True))
#             corrected_form_and_compounds_per_sentence_and = data_utils.load_corrected_form_compounds_per_sentence_and()
#             get_compound_and_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_and_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compound_and', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)


#         if "compound_but" in representations:

#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors = 'pt', pad_to_max_length=True))
#             corrected_form_but_compounds_per_sentence_but = data_utils.load_corrected_form_compounds_per_sentence_and()
#             get_compound_but_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_but_compounds_per_sentence_but, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compound_but', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)

#         if "compound_that" in representations:

#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors = 'pt', pad_to_max_length=True))
#             corrected_form_that_compounds_per_sentence_that = data_utils.load_corrected_form_compounds_per_sentence_that()
#             get_compound_that_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_that_compounds_per_sentence_that, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compound_that', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)



# #
#         if "full_stop" in representations:
#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
#             get_cls_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds_full_stop', rep_loc=rep_loc,torch_device=torch_device)

#         if "compositional_analysis_and" in representations:
#             input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(comp_sentences, max_length=512, return_tensors='pt', pad_to_max_length=True))
#             get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, words_to_find, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compositional_analysis_and', torch_device=torch_device, save_attention=save_attention, rep_loc=rep_loc)
        
#         else:
#             print('No representations specified')

# if __name__ == "__main__":
#     args = parser.parse_args()
#     print(vars(args))

#     extract_and_save_representations(amount_of_dataset=args.amount_of_dataset, load_if_available=args.load_if_available, batch_size=args.batch_size, models=args.models, layers=args.layers, torch_device=args.device, representations=args.representations, rep_loc=args.rep_loc, save_attention=args.save_attention)


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
parser.add_argument('--representations', nargs='+', type=str, default=None, help='Representations to extract ("cls", "mean_pooled", "noun_noun_compound")')
parser.add_argument('--rep_loc', type=str, default="./data", help='Where to save representations')
parser.add_argument('--save_attention', dest='save_attention', action='store_true', default=False)
parser.add_argument('--load_if_available', dest='load_if_available', action='store_true', default=False)
parser.add_argument('--amount_of_dataset', type=float, default=1.0, help='Proportion of dataset to use')


def get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, load_if_available=True, batch_size = 1, rep_loc='./data', rep_type="mean_pooled", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=mean_pool_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, save_attention=save_attention)
     
def get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="noun_noun_compound", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=noun_noun_compound_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, middle_dim=2, save_attention=save_attention)

def get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, word_per_sentence, layers, load_if_available=True, batch_size = 1, rep_type="compositional_analysis", rep_loc='./data', torch_device="cuda", save_attention=False):
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

def single_word_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, word_per_sentence):
    # Get tokens where tokens aren't special tokens or pad tokens

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/openai-gpt', 'openai-community/gpt2', 'openai-community/gpt2-xl', "meta-llama/Llama-3.1-8B"]:
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
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/openai-gpt', 'openai-community/gpt2', 'openai-community/gpt2-xl', "meta-llama/Llama-3.1-8B"]:
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

def mean_pool_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i=None):
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    pad_token_mask = lambda x: np.array(x.cpu() == tokeniser.pad_token_id)
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x) * (pad_token_mask(x) == False)).reshape(-1)
    
    
    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/openai-gpt', 'openai-community/gpt2', 'openai-community/gpt2-xl', "meta-llama/Llama-3.1-8B"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]

    
    return np.vstack([np.mean(reps[get_tokens_to_keep(input_ids[i])].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])



def get_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, token_selector=mean_pool_selector, load_if_available=True, batch_size=1, rep_type="sentence_pair_cls", torch_device="cuda", save_reps=True, data_loc='./data', add_arg_dict={}, middle_dim=None, save_attention=False):
    '''
    layer 0: embedding
    layer 1: first layer
    . . . 
    layer 12: usually the final layer
    NOTE: Would probably be much faster to extract all layers at the same time (assuming we want outputs from all layers)
    '''
    rep_locs_per_layer = [data_utils.get_hidden_state_file(model_name, layer=x, rep_type=rep_type, data_loc=data_loc) for x in layers]
    load_reps = load_if_available and os.path.isfile(rep_loc)

    layer_map = dict(zip(layers, range(len(layers))))
    layer1=[]
    layer2=[]
    layer3=[]
    layer4=[]
    layer5=[]
    layer6=[]
    layer7=[]
    layer8=[]
    layer9=[]
    layer10=[]
    layer11=[]
    layer12=[]
    layer13=[]
    layer14=[]
    layer15=[]
    layer16=[]
    layer17=[]
    layer18=[]
    layer19=[]
    layer20=[]
    layer21=[]
    layer22=[]
    layer23=[]
    layer24=[] 

    if load_reps:
        tokens_per_layer = [np.load(x) for x in rep_locs_per_layer]
    else:
        print('Extracting representations from model for layers {}'.format(layers))
        
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        model.to(torch_device)
        
        ## remove the 2 when using mean_pool_selector, only noun_noun_compound_selector uses it
        if middle_dim == 2:
            tokens_per_layer = [np.zeros((input_ids.shape[0], 2, model.config.hidden_size)) for x in layers]
        else:
            tokens_per_layer = [np.zeros((input_ids.shape[0], model.config.hidden_size)) for x in layers]
            print(len(tokens_per_layer))
        
        if save_attention:
            # Calculate size of attention matrix
            non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
            pad_token_mask = lambda x: np.array(x.cpu() == tokeniser.pad_token_id)
            get_tokens_to_keep = lambda x: np.argwhere((pad_token_mask(x) == False)).reshape(-1)
            tokens_to_keep_per_sample = [get_tokens_to_keep(x) for x in input_ids]
            seq_lens = np.array([len(x) for x in tokens_to_keep_per_sample])
            seq_len = max(seq_lens)
            attention_per_layer = [-np.ones((input_ids.shape[0], seq_len**2)) for x in layers]

            decode_tokens = lambda x: [tokeniser.decode(token) for token in x[get_tokens_to_keep(x)].tolist()]
            '\n'.join([''.join(decode_tokens(x)) for x in input_ids])

        with torch.no_grad():
            # iterate from 0 to 900
            for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):

                outputs = model(input_ids[i:i+batch_size, :].reshape(batch_size, -1), attention_mask=attention_mask[i:i+batch_size, :].reshape(batch_size, -1), output_attentions=save_attention, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                              
                
                add_arg_dict["i"] = i



                for layer in layers:
                   
                    tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = token_selector(model, model_name, tokeniser, hidden_states, input_ids[i:i+batch_size, :].reshape(batch_size, -1), layer, batch_size, **add_arg_dict)
                    


                    if (layer)%24 == 1:
                        layer1.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 2:
                        layer2.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 3:
                        layer3.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 4:
                        layer4.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 5:
                        layer5.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 6:
                        layer6.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 7:
                        layer7.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 8:
                        layer8.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 9:
                        layer9.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 10:
                        layer10.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 11:
                        layer11.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 12:
                        layer12.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 13:
                        layer13.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 14:
                        layer14.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 15:
                        layer15.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 16:
                        layer16.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 17:
                        layer17.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 18:
                        layer18.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 19:
                        layer19.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 20:
                        layer20.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 21:
                        layer21.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 22:
                        layer22.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 23:
                        layer23.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    elif (layer)%24 == 0:
                        layer24.append(tokens_per_layer[layer_map[layer]][i:i+batch_size, :])
                    
                   
        layer1 = np.array(layer1)
        layer2 = np.array(layer2)
        layer3 = np.array(layer3)
        layer4 = np.array(layer4)
        layer5 = np.array(layer5)
        layer6 = np.array(layer6)
        layer7 = np.array(layer7)
        layer8 = np.array(layer8)
        layer9 = np.array(layer9)
        layer10 = np.array(layer10)
        layer11 = np.array(layer11)
        layer12 = np.array(layer12)
        layer13 = np.array(layer13)
        layer14 = np.array(layer14)
        layer15 = np.array(layer15)
        layer16 = np.array(layer16)
        layer17 = np.array(layer17)
        layer18 = np.array(layer18)
        layer19 = np.array(layer19)
        layer20 = np.array(layer20)
        layer21 = np.array(layer21)
        layer22 = np.array(layer22)
        layer23 = np.array(layer23)
        layer24 = np.array(layer24)

        layer1 = layer1.reshape(900, 2048)
        layer2 = layer2.reshape(900, 2048)
        layer3 = layer3.reshape(900, 2048)
        layer4 = layer4.reshape(900, 2048)
        layer5 = layer5.reshape(900, 2048)
        layer6 = layer6.reshape(900, 2048)
        layer7 = layer7.reshape(900, 2048)
        layer8 = layer8.reshape(900, 2048)
        layer9 = layer9.reshape(900, 2048)
        layer10 = layer10.reshape(900, 2048)
        layer11 = layer11.reshape(900, 2048)
        layer12 = layer12.reshape(900, 2048)
        layer13 = layer13.reshape(900, 2048)
        layer14 = layer14.reshape(900, 2048)
        layer15 = layer15.reshape(900, 2048)
        layer16 = layer16.reshape(900, 2048)
        layer17 = layer17.reshape(900, 2048)
        layer18 = layer18.reshape(900, 2048)
        layer19 = layer19.reshape(900, 2048)
        layer20 = layer20.reshape(900, 2048)
        layer21 = layer21.reshape(900, 2048)
        layer22 = layer22.reshape(900, 2048)
        layer23 = layer23.reshape(900, 2048)
        layer24 = layer24.reshape(900, 2048)

        

        np.save("phi-1_layer_1_noun_noun_mean_pooled.npy", layer1)
        np.save("phi-1_layer_2_noun_noun_mean_pooled.npy", layer2)
        np.save("phi-1_layer_3_noun_noun_mean_pooled.npy", layer3)
        np.save("phi-1_layer_4_noun_noun_mean_pooled.npy", layer4)
        np.save("phi-1_layer_5_noun_noun_mean_pooled.npy", layer5)
        np.save("phi-1_layer_6_noun_noun_mean_pooled.npy", layer6)
        np.save("phi-1_layer_7_noun_noun_mean_pooled.npy", layer7)
        np.save("phi-1_layer_8_noun_noun_mean_pooled.npy", layer8)
        np.save("phi-1_layer_9_noun_noun_mean_pooled.npy", layer9)
        np.save("phi-1_layer_10_noun_noun_mean_pooled.npy", layer10)
        np.save("phi-1_layer_11_noun_noun_mean_pooled.npy", layer11)
        np.save("phi-1_layer_12_noun_noun_mean_pooled.npy", layer12)
        np.save("phi-1_layer_13_noun_noun_mean_pooled.npy", layer13)
        np.save("phi-1_layer_14_noun_noun_mean_pooled.npy", layer14)
        np.save("phi-1_layer_15_noun_noun_mean_pooled.npy", layer15)
        np.save("phi-1_layer_16_noun_noun_mean_pooled.npy", layer16)
        np.save("phi-1_layer_17_noun_noun_mean_pooled.npy", layer17)
        np.save("phi-1_layer_18_noun_noun_mean_pooled.npy", layer18)
        np.save("phi-1_layer_19_noun_noun_mean_pooled.npy", layer19)
        np.save("phi-1_layer_20_noun_noun_mean_pooled.npy", layer20)
        np.save("phi-1_layer_21_noun_noun_mean_pooled.npy", layer21)
        np.save("phi-1_layer_22_noun_noun_mean_pooled.npy", layer22)
        np.save("phi-1_layer_23_noun_noun_mean_pooled.npy", layer23)
        np.save("phi-1_layer_24_noun_noun_mean_pooled.npy", layer24)
    
        


    if save_attention:
        for layer in layers:
            atten_rep_loc = data_utils.get_hidden_state_file(model_name, layer=layer, rep_type=rep_type + '_attention', data_loc=data_loc)
            pathlib.Path('/'.join(atten_rep_loc.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            np.save(atten_rep_loc, attention_per_layer[layer_map[layer]])

        return tokens_per_layer, attention_per_layer
    else:
               
        return tokens_per_layer




def extract_and_save_representations(amount_of_dataset=1, batch_size=1, models=None, load_if_available=False, layers=None, torch_device="cpu", representations=["mean_pooled"], rep_loc='./data', save_attention=False):

    if representations == None:
        representations = ["mean_pooled"]

    sentences = data_utils.get_noun_noun_compound_sentences()
    mod_head_words_per_sentence = data_utils.get_noun_noun_mod_head_words_per_sentence()

    num_to_keep = int(amount_of_dataset * len(sentences))
    sentences = sentences[:num_to_keep]
    mod_head_words_per_sentence = mod_head_words_per_sentence[:num_to_keep]

    if "compositional_analysis" in representations:
        words_to_find, comp_sentences = data_utils.get_compositional_probe_words_and_sentences()
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
        if "noun_noun_compounds" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_corrected_form_compounds_per_sentence()
            get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='noun_noun_compounds', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)

        if "compositional_analysis" in representations:
            inputs = tokeniser(comp_sentences,  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(comp_sentences, max_length=512, return_tensors='pt', pad_to_max_length=True))
            get_word_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, words_to_find, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='compositional_analysis', torch_device=torch_device, save_attention=save_attention, rep_loc=rep_loc)
        
        else:
            print('No representations specified')

if __name__ == "__main__":
    args = parser.parse_args()
    print(vars(args))

    extract_and_save_representations(amount_of_dataset=args.amount_of_dataset, load_if_available=args.load_if_available, batch_size=args.batch_size, models=args.models, layers=args.layers, torch_device=args.device, representations=args.representations, rep_loc=args.rep_loc, save_attention=args.save_attention)