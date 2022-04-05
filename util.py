#!/usr/bin/env python
"""Provides FileReader, class for main.

FileReader defines several file reading operations.
"""

# standard library
import os
import csv

# 3rd party packages
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tag import CRFTagger

__author__ = "Samuel Situmeang"
__copyright__ = "Copyright 2022, Institut Teknologi Del"
__credits__ = ["Samuel Situmeang"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Samuel Situmeang"
__email__ = "samuel.situmeang@del.ac.id"
__status__ = "Production"

class FileReader:
    """FileReader class."""
    def __init__(self, fname):
        self.fname = fname
    
    def check_type(self):
        """FileReader.check_type() should return the file extension if only the file is exists."""
        file_exists = os.path.exists(self.fname)
        if file_exists:
            print(f"The file {self.fname} exists.")
            ext = os.path.splitext(self.fname)[-1].lower()
            return ext
        else:
            print(f'The file {self.fname} does not exist.')
        
    def read_tsv(self):
        """FileReader.read_tsv() should return the data in a DataFrame."""
        try:
            df = pd.read_csv(self.fname,
                            sep="\t",
                            names=["token", "ne"],
                            skip_blank_lines=False,
                            quoting=csv.QUOTE_NONE,
                            encoding='utf-8')
            
            list_tmp = []
            no = 0
            for row in df.itertuples():
                if pd.isnull(row.token):
                    list_tmp.append(np.nan)
                    no+=1
                else:
                    list_tmp.append(no)
            df["sentence"] = list_tmp
            df = df.dropna(thresh=2).reset_index(drop=True)
            df[["sentence"]] = df[["sentence"]].astype(int)

            return df
        except Exception as e:
            print(e)

class DatasetPreparator:
    """DatasetPreparator class."""
    def __init__(self, df, pt):
        self.df = df
        self.pt = pt
    
    def add_post(self):
        list_post = []

        ct = CRFTagger()
        ct.set_model_file(self.pt)
        series_tmp = self.df.groupby("sentence")["token"].apply(list)
        df_tmp = series_tmp.to_frame(name="token")
        for val in tqdm(df_tmp.itertuples(), total=df_tmp.shape[0]):
            post_tag = ct.tag_sents([val.token])
            list_post += [e[1] for e in post_tag[0]]
        self.df["post"] = list_post

    def check_post(self):
        """DatasetPreparator.check_post() should return True only if pos-tag column exist."""
        if ("post" or "postag") in self.df.columns:
            print(f"\nThe pos tag column exists.")
        else:
            print(f"\nThe pos tag column does not exist.")
            print(f"Creating column in progress....")
            self.add_post()
            self.df = self.df[["sentence", "token", "post", "ne"]]
            return self.df
            
    def add_bio_ne(self):
        dfd = self.df.copy()
        bio_tag = []
        prev_tag = "O"
        for _, tag in self.df["ne"].iteritems():
            if tag == "O": #O
                bio_tag.append((tag))
                prev_tag = tag
                continue
            if tag != "O" and prev_tag == "O": # Begin NE
                bio_tag.append(("B-"+tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag == tag: # Inside NE
                bio_tag.append(("I-"+tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag != tag: # nearby NE
                bio_tag.append(("B-"+tag))
                prev_tag = tag
        
        dfd["bio_ne"] = bio_tag
        self.df = dfd
        return self.df

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['token'].values.tolist(), 
                                                           s['post'].values.tolist(), 
                                                           s['bio_ne'].values.tolist())]
        self.grouped = self.data.groupby('sentence').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

# Feature set
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for _, _, label in sent]
    
def sent2tokens(sent):
    return [token for token, _, _ in sent]