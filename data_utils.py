# -*- coding: utf-8 -*-
# file: data_utils.py
# basecode: songyouwei <youwei0314@gmail.com>
# author: xumayi <xumayi@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx
import spacy
from pytorch_transformers import BertTokenizer,XLNetTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Pretrain:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    # Group distance to aspect of an original word to its corresponding subword token
    def tokenize(self, text, dep_dist, reverse=False, padding='post', truncating='post'):
        sequence, distances = [],[]
        for word,dist in zip(text,dep_dist):
            tokens = self.tokenizer.tokenize(word)
            for jx,token in enumerate(tokens):
                sequence.append(token)
                distances.append(dist)
        sequence = self.tokenizer.convert_tokens_to_ids(sequence)

        if len(sequence) == 0:
            sequence = [0]
            dep_dist = [0]
        if reverse:
            sequence = sequence[::-1]
            dep_dist = dep_dist[::-1]
        sequence = pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        dep_dist = pad_and_truncate(dep_dist, self.max_seq_len, padding=padding, truncating=truncating,value=self.max_seq_len)

        return sequence, dep_dist


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left_temp, _, text_right_temp = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect_temp = lines[i + 1].lower().strip()

            text_left = ''
            text_right =''
            aspect = ''

            text_left_list = nlp(text_left_temp)
            for token in text_left_list:
                text_left = text_left + ' ' + token.lower_
            text_left = text_left.strip()

            text_right_list = nlp(text_right_temp)
            for token in text_right_list:
                text_right = text_right + ' ' + token.lower_
            text_right = text_right.strip()

            aspect_list = nlp(aspect_temp)
            for token in aspect_list:
                aspect = aspect + ' ' + token.lower_
            aspect = aspect.strip()

            auxiliary_aspect = 'What is the polarity of {}'.format(aspect)
            polarity = lines[i + 2].strip()

            raw_text = text_left + " " + aspect + " " + text_right

            depend_word, depended_word, depend, depended, no_connect = cluster_calculate(raw_text, aspect)

            depend = np.array(pad_and_truncate(depend, tokenizer.max_seq_len,value = -1))

            depended = np.array(pad_and_truncate(depended, tokenizer.max_seq_len, value = -1))

            no_connect = np.array(pad_and_truncate(no_connect, tokenizer.max_seq_len, value = -1))

            text_raw_indices = tokenizer.text_to_sequence(raw_text)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            auxiliary_aspect_indices = tokenizer.text_to_sequence(auxiliary_aspect)
            auxiliary_aspect_len = np.sum(auxiliary_aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1
            sent = text_left + " " + aspect + " " + text_right
            text_bert_indices = tokenizer.text_to_sequence(tokenizer.cls_token+' ' + sent + ' '
                                                           +tokenizer.sep_token+' ' + aspect + " " +tokenizer.sep_token)

            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            if 'Roberta' in type(tokenizer.tokenizer).__name__:
                bert_segments_ids = np.zeros(np.sum(text_raw_indices != 0) + 2 + aspect_len + 1)
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence(tokenizer.cls_token+ ' ' + text_left + " " + aspect + " " + text_right + " " + tokenizer.sep_token)
            text_raw_bert_indices_len = np.sum( text_raw_bert_indices != 0)
            # Find distance in dependency parsing tree
            raw_tokens, dist , max_dist = calculate_dep_dist(sent,aspect)
            raw_tokens.insert(0,tokenizer.cls_token)
            dist.insert(0,0)
            raw_tokens.append(tokenizer.sep_token)
            dist.append(0)

            _, distance_to_aspect = tokenizer.tokenize(raw_tokens, dist)
            aspect_bert_indices = tokenizer.text_to_sequence(tokenizer.cls_token+ ' ' + aspect + " " + tokenizer.sep_token)

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'text_raw_bert_indices_len': text_raw_bert_indices_len,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
                'dep_distance_to_aspect':distance_to_aspect,
                'raw_text':raw_text,
                'aspect':aspect,
                'depend':depend,
                'depended':depended,
                'no_connect':no_connect,
                'max_dist':max_dist,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
nlp = spacy.load("en_core_web_sm")

def calculate_dep_dist(sentence,aspect):
    terms = [a.lower() for a in aspect.split()]
    doc = nlp(sentence)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    cnt = 0
    term_ids = [0] * len(terms)
    for token in doc:
        # Record the position of aspect terms
        if cnt < len(terms) and token.lower_ == terms[cnt]:
            term_ids[cnt] = token.i
            cnt += 1

        for child in token.children:
            edges.append(('{}_{}'.format(token.lower_,token.i),
                          '{}_{}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)

    dist = [0.0]*len(doc)
    text = [0]*len(doc)
    max_dist_temp = []
    for i,word in enumerate(doc):
        source = '{}_{}'.format(word.lower_,word.i)
        sum = 0
        flag = 1
        max_dist = 0
        for term_id,term in zip(term_ids,terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph,source=source,target=target)
            except:
                sum += len(doc) # No connection between source and target
                flag = 0
        dist[i] = sum/len(terms)
        text[i] = word.text
        if flag == 1:
            max_dist_temp.append(sum/len(terms))
        if dist[i] > max_dist:
            max_dist = dist[i]
   
    return text,dist,max_dist

def cluster_calculate(sentence, aspect):
    terms = [a.lower() for a in aspect.split()]
    
    doc_list = []
    doc = [a.lower() for a in sentence.split()]
    for i in range(len(doc)):
        doc_list.append(i)

    doc = nlp(sentence.strip())
    # Load spacy's dependency tree into a networkx graph
    edges = []
    cnt = 0
    term_ids = [0] * len(terms)
    for token in doc:
        # Record the position of aspect terms
        if cnt < len(terms) and token.lower_ == terms[cnt]:
            term_ids[cnt] = token.i
            cnt += 1

        for child in token.children:
            edges.append((token.i,child.i))
                  
    graph = nx.DiGraph(edges)
    graph2 = nx.Graph(edges)
    
    
    no_connect = []
    for i,word in enumerate(doc):
        source = i
        for j in term_ids:
            target = j
            try:
                sum = nx.shortest_path_length(graph2,source=source,target=target)
            except:
                if (i not in no_connect) and (i not in term_ids):
                    no_connect.append(i)           
                   
    depend_ids = []
    depended_ids = doc_list
    for k in range(len(terms)):
        temp_aspcet_ids = term_ids[k];
        try:
            temp_nodes = list(nx.dfs_preorder_nodes(graph, source=temp_aspcet_ids))
        except:
            temp_nodes = [temp_aspcet_ids]
            
        for i in range(len(temp_nodes)):           
                flag = 1
                for j in range(len(depend_ids)):
                    if depend_ids[j] == temp_nodes[i]:
                        flag = 0
                if flag == 1:
                    depend_ids.append(temp_nodes[i])

    for i in range(len(depend_ids)):
        s=depend_ids[i]
        depended_ids.remove(s)
        
    for i in range(len(terms)):
        temp_aspcet_ids = term_ids[i]
        if temp_aspcet_ids in depend_ids:
            depend_ids.remove(temp_aspcet_ids)
    
    for i in range(len(terms)):
        temp_aspcet_ids = term_ids[i]
        if temp_aspcet_ids in depended_ids:
            depended_ids.remove(temp_aspcet_ids)
        
    depend_ids.sort()
    depended_ids.sort()
    
    for i in range(len(no_connect)):
        if no_connect[i] in depended_ids:
            depended_ids.remove(no_connect[i])

    depend_word = ''
    depended_word = ''
    for i in range(len(depend_ids)):
        depend_word = depend_word + str(doc[depend_ids[i]]) + ' '
    for i in range(len(depended_ids)):
        depended_word = depended_word + str(doc[depended_ids[i]]) + ' '

    depend_word = depend_word.strip()
    depended_word = depended_word.strip()
    return depend_word,depended_word,depend_ids,depended_ids,no_connect