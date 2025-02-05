import pandas as pd
import os
import numpy as np
import tqdm
import torch

import rsa_utils

# def load_corrected_form_compounds_per_sentence(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_csv('{}/noun_noun_compounds/correct_forms_copy.csv'.format(data_loc))
#     return list(zip(df.mod_word_match.tolist(), df.head_noun_match.tolist()))

def load_corrected_form_compounds_per_sentence(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_csv('{}/noun_noun_compounds/correct_forms_copy.csv'.format(data_loc))
    return list(zip(df.mod_word_match.tolist(), df.head_noun_match.tolist()))


# def load_corrected_form_compounds_per_sentence_and(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_csv('{}/noun_noun_compounds/correct_forms_and.csv'.format(data_loc))
#     return list(zip(df.head_noun_match.tolist(), df.and_word_match.tolist()))

def load_corrected_form_compounds_per_sentence_and(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_csv('{}/noun_noun_compounds/correct_forms_and.csv'.format(data_loc))
    return list(zip(df.head_noun_match.tolist(), df.and_word_match.tolist()))


# def load_corrected_form_compounds_per_sentence_but(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_csv('{}/noun_noun_compounds/correct_forms_but.csv'.format(data_loc))
#     return list(zip(df.head_noun_match.tolist(), df.but_word_match.tolist()))

def load_corrected_form_compounds_per_sentence_but(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_csv('{}/noun_noun_compounds/correct_forms_but.csv'.format(data_loc))
    return list(zip(df.head_noun_match.tolist(), df.but_word_match.tolist()))

# def load_corrected_form_compounds_per_sentence_that(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_csv('{}/noun_noun_compounds/correct_forms_that.csv'.format(data_loc))
#     return list(zip(df.head_noun_match.tolist(), df.that_word_match.tolist()))

def load_corrected_form_compounds_per_sentence_that(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_csv('{}/noun_noun_compounds/correct_forms_that.csv'.format(data_loc))
    return list(zip(df.head_noun_match.tolist(), df.that_word_match.tolist()))






# def get_hidden_state_file(model_name, layer=11, rep_type='sentence_pair_cls', data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     hidden_state_folder = '{}/representations/{}/layer_{}/{}'.format(data_loc, model_name.split('-')[0], layer, rep_type)
#     return '{}/{}_layer_{}_{}.npy'.format(hidden_state_folder, model_name, layer, rep_type)


def get_hidden_state_file(model_name, layer=11, rep_type='sentence_pair_cls', data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    hidden_state_folder = '{}/representations/{}/layer_{}/{}'.format(data_loc, model_name.split('-')[0], layer, rep_type)
    return '{}/{}_layer_{}_{}.npy'.format(hidden_state_folder, model_name, layer, rep_type)


# def get_noun_noun_compound_sentences(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_copy.xlsx'.format(data_loc), skiprows=9)
#     df = df[~df["NN sentence"].isnull()]
#     sentences = np.array(df['NN sentence'][:300].tolist() + df['Gloss sentence'].tolist())

#     return sentences

def get_noun_noun_compound_sentences(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_copy.xlsx'.format(data_loc), skiprows=9)
    df = df[~df["NN sentence"].isnull()]
    sentences = np.array(df['NN sentence'][:300].tolist() + df['Gloss sentence'].tolist())

    return sentences

# def get_noun_noun_and_compounds_sentences(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
#     #df = df[~df["NN sentence"].isnull()]
#     sentences = np.array(df['AND_sentence'][:300].tolist() + df['Gloss sentence AND'].tolist())

#     return sentences

def get_noun_noun_and_compounds_sentences(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
    #df = df[~df["NN sentence"].isnull()]
    sentences = np.array(df['AND_sentence'][:300].tolist() + df['Gloss sentence AND'].tolist())

    return sentences

# def get_noun_noun_but_compounds_sentences(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
#     #df = df[~df["NN sentence"].isnull()]
#     sentences = np.array(df['BUT_sentence'][:300].tolist() + df['Gloss sentence BUT'].tolist())

#     return sentences

def get_noun_noun_but_compounds_sentences(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
    #df = df[~df["NN sentence"].isnull()]
    sentences = np.array(df['BUT_sentence'][:300].tolist() + df['Gloss sentence BUT'].tolist())

    return sentences

# def get_noun_noun_that_compounds_sentences(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
#     #df = df[~df["NN sentence"].isnull()]
#     sentences = np.array(df['THAT_sentence'][:300].tolist() + df['Gloss sentence THAT'].tolist())

#     return sentences

def get_noun_noun_that_compounds_sentences(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
    #df = df[~df["NN sentence"].isnull()]
    sentences = np.array(df['THAT_sentence'][:300].tolist() + df['Gloss sentence THAT'].tolist())

    return sentences

# def get_noun_noun_mod_head_words_per_sentence(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_copy.xlsx'.format(data_loc), skiprows=9)
#     df = df[~df["NN sentence"].isnull()]
#     mod_head_tuples_per_sentence = np.array(list(zip(df['mod'][:300].tolist(), df['head'][:300].tolist())))
#     all_mod_head_tuples_per_sentence = np.vstack([mod_head_tuples_per_sentence, mod_head_tuples_per_sentence, mod_head_tuples_per_sentence])

#     return all_mod_head_tuples_per_sentence

def get_noun_noun_mod_head_words_per_sentence(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_copy.xlsx'.format(data_loc), skiprows=9)
    df = df[~df["NN sentence"].isnull()]
    mod_head_tuples_per_sentence = np.array(list(zip(df['mod'][:300].tolist(), df['head'][:300].tolist())))
    all_mod_head_tuples_per_sentence = np.vstack([mod_head_tuples_per_sentence, mod_head_tuples_per_sentence, mod_head_tuples_per_sentence])

    return all_mod_head_tuples_per_sentence

# def get_noun_noun_head_and_words_per_sentence(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
#     df = df[~df["NN sentence"].isnull()]
#     head_and_tuples_per_sentence = np.array(list(zip(df['head'][:300].tolist(), df['and'][:300].tolist())))
#     all_head_and_tuples_per_sentence = np.vstack([head_and_tuples_per_sentence, head_and_tuples_per_sentence, head_and_tuples_per_sentence])

#     return all_head_and_tuples_per_sentence

def get_noun_noun_head_and_words_per_sentence(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
    df = df[~df["NN sentence"].isnull()]
    head_and_tuples_per_sentence = np.array(list(zip(df['head'][:300].tolist(), df['and'][:300].tolist())))
    all_head_and_tuples_per_sentence = np.vstack([head_and_tuples_per_sentence, head_and_tuples_per_sentence, head_and_tuples_per_sentence])

    return all_head_and_tuples_per_sentence

# def get_noun_noun_head_but_words_per_sentence(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
#     df = df[~df["NN sentence"].isnull()]
#     head_but_tuples_per_sentence = np.array(list(zip(df['head'][:300].tolist(), df['but'][:300].tolist())))
#     all_head_but_tuples_per_sentence = np.vstack([head_but_tuples_per_sentence, head_but_tuples_per_sentence, head_but_tuples_per_sentence])

#     return all_head_but_tuples_per_sentence

def get_noun_noun_head_but_words_per_sentence(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
    df = df[~df["NN sentence"].isnull()]
    head_but_tuples_per_sentence = np.array(list(zip(df['head'][:300].tolist(), df['but'][:300].tolist())))
    all_head_but_tuples_per_sentence = np.vstack([head_but_tuples_per_sentence, head_but_tuples_per_sentence, head_but_tuples_per_sentence])

    return all_head_but_tuples_per_sentence

# def get_noun_noun_head_that_words_per_sentence(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data'):
#     df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
#     df = df[~df["NN sentence"].isnull()]
#     head_that_tuples_per_sentence = np.array(list(zip(df['head'][:300].tolist(), df['that'][:300].tolist())))
#     all_head_that_tuples_per_sentence = np.vstack([head_that_tuples_per_sentence, head_that_tuples_per_sentence, head_that_tuples_per_sentence])

#     return all_head_that_tuples_per_sentence

def get_noun_noun_head_that_words_per_sentence(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data'):
    df = pd.read_excel('{}/noun_noun_compounds/nn_compounds_with_continuation_words1.xlsx'.format(data_loc))
    df = df[~df["NN sentence"].isnull()]
    head_that_tuples_per_sentence = np.array(list(zip(df['head'][:300].tolist(), df['that'][:300].tolist())))
    all_head_that_tuples_per_sentence = np.vstack([head_that_tuples_per_sentence, head_that_tuples_per_sentence, head_that_tuples_per_sentence])

    return all_head_that_tuples_per_sentence




#def get_sts_b_dataframe(amount_of_dataset=1, sts_b_loc='./data/stsbenchmark'):
    # Load STS-Benchmark dataset
    sts_b_train, sts_b_dev, sts_b_test = get_sts_b_data(sts_b_loc)

    sts_b_data = pd.concat([sts_b_train, sts_b_dev, sts_b_test])

    # If amount_of_dataset < 1, truncate dataset (for development; TODO: shuffle first?)
    to_keep = int(sts_b_data.shape[0] * amount_of_dataset)
    sts_b_data = sts_b_data[:to_keep]
    if amount_of_dataset < 1:
        print('amount_of_dataset < 1 ({})\tUsing {} samples'.format(amount_of_dataset, to_keep))

    return sts_b_data

#def load_all_sts_b_data(amount_of_dataset=1, sts_b_loc='./data/stsbenchmark'):
    sts_b_data = get_sts_b_dataframe(amount_of_dataset=amount_of_dataset, sts_b_loc=sts_b_loc)

    # Extract sentence pairs and STS scores
    sentence_pairs = list(zip(sts_b_data.sentence1.tolist(), sts_b_data.sentence2.tolist()))
    sts_b_scores = sts_b_data.score.to_numpy()

    return sentence_pairs, sts_b_scores

#def fix_sts_b_encoding_issues(sts_b_df):
    char_map = {'‚Äô': "'", '‚Äė': "'", 'â€™': "'", "‚Äď": "—", "¿": "'", " ŔÄ": ""}
            
    for symbol in char_map.keys():
        sts_b_df.sentence1 = sts_b_df.sentence1.str.replace(symbol, char_map[symbol])
        sts_b_df.sentence2 = sts_b_df.sentence2.str.replace(symbol, char_map[symbol])

    return sts_b_df

#def get_sts_b_data(sts_b_loc='./data/stsbenchmark'):
    column_names = ['genre', 'filename', 'year', 'id', 'score', 'sentence1', 'sentence2']
    load_df = lambda x: pd.read_csv('{}/{}.csv'.format(sts_b_loc, x), delimiter='\t', header=None, names=column_names, quoting=3, escapechar="\\")
    
    sts_b_train, sts_b_dev, sts_b_test = [fix_sts_b_encoding_issues(load_df(x)) for x in ['sts-train', 'sts-dev', 'sts-test']]

    return sts_b_train, sts_b_dev, sts_b_test





# def get_compositional_probe_words_and_sentences(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

#     rows = [x for x in open('{}/probe_words_and_sentences_300_copy.txt'.format(data_loc)).read().split('\n') if x != '']
#     words = [x.split('\t')[0] for x in rows]
#     sentences = [x.split('\t')[-1] for x in rows]
#     return words, rows

def get_compositional_probe_words_and_sentences(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

    rows = [x for x in open('{}/probe_words_and_sentences_300_copy.txt'.format(data_loc)).read().split('\n') if x != '']
    words = [x.split('\t')[0] for x in rows]
    sentences = [x.split('\t')[-1] for x in rows]
    return words, rows



# def get_compositional_probe_words_and_sentences_and(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

#     rows = [x for x in open('{}/probe_words_and_sentences_300_with_and.txt'.format(data_loc)).read().split('\n') if x != '']
#     words = [x.split('\t')[0] for x in rows]
#     sentences = [x.split('\t')[-1] for x in rows]
#     return words, rows


def get_compositional_probe_words_and_sentences_and(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

    rows = [x for x in open('{}/probe_words_and_sentences_300_with_and.txt'.format(data_loc)).read().split('\n') if x != '']
    words = [x.split('\t')[0] for x in rows]
    sentences = [x.split('\t')[-1] for x in rows]
    return words, rows

# def get_compositional_probe_words_and_sentences_but(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

#     rows = [x for x in open('{}/probe_words_and_sentences_300_with_but.txt'.format(data_loc)).read().split('\n') if x != '']
#     words = [x.split('\t')[0] for x in rows]
#     sentences = [x.split('\t')[-1] for x in rows]
#     return words, rows

def get_compositional_probe_words_and_sentences_but(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

    rows = [x for x in open('{}/probe_words_and_sentences_300_with_but.txt'.format(data_loc)).read().split('\n') if x != '']
    words = [x.split('\t')[0] for x in rows]
    sentences = [x.split('\t')[-1] for x in rows]
    return words, rows

# def get_compositional_probe_words_and_sentences_that(data_loc='/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

#     rows = [x for x in open('{}/probe_words_and_sentences_300_with_that.txt'.format(data_loc)).read().split('\n') if x != '']
#     words = [x.split('\t')[0] for x in rows]
#     sentences = [x.split('\t')[-1] for x in rows]
#     return words, rows


def get_compositional_probe_words_and_sentences_that(data_loc='E:/NOUN-NOUN-COMPOUNDS-V1/data/noun_noun_compounds/composition'):

    rows = [x for x in open('{}/probe_words_and_sentences_300_with_that.txt'.format(data_loc)).read().split('\n') if x != '']
    words = [x.split('\t')[0] for x in rows]
    sentences = [x.split('\t')[-1] for x in rows]
    return words, rows




def select_within_compound_groups(rdm, group_i, within_compound_sentences=False):
    to_keep_inds = []
    to_keep_inds_within_compound_sentences = []

    get_lower = lambda x: x[np.where(np.triu(np.ones(x.shape[:1])) == 0)]
    
    for start in list(range(0, 900, 15)):
        block_inds = [[(i, j) for i in range(start, start + 15)] for j in range(start, start + 15)]
        to_keep_inds.append(get_lower(np.array(block_inds)))
        
        within_sentence_inds = np.arange(15) % 3 == 0
        to_keep_inds_within_compound_sentences.append(get_lower(np.array(block_inds)[within_sentence_inds, :][:, within_sentence_inds]))

    if not within_compound_sentences:
        return np.array([rdm[i[0]][i[1]] for i in to_keep_inds[group_i]])
    else:
        return np.array([rdm[i[0]][i[1]] for i in to_keep_inds_within_compound_sentences[group_i]])

#if __name__ == "__main__":
    load_all_sts_b_data()