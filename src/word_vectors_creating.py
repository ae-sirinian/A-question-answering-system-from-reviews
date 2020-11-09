#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:33:54 2020

@author: aram
"""
from gensim.models import KeyedVectors
import sys
from config import WV_FILE_PATH
from config import GOOGLE_WORD2VEC_MODEL_FILE_PATH

def create_word_vectors_with_word2vec(load_model = False,
                        wv_file_path = WV_FILE_PATH,
                        ram_limit = 100000,
                        google_word2vec_model_file_path =
                        GOOGLE_WORD2VEC_MODEL_FILE_PATH):
    
    if load_model == True:
        try:
            model = KeyedVectors.load_word2vec_format(
                    google_word2vec_model_file_path,
                    binary = True, limit = ram_limit)
        except FileNotFoundError:
            print("Error while loading google's word2vec model file: ", 
                  "google's word2vec model file can't be found in ", 
                  google_word2vec_model_file_path, " file path!")
            sys.exit()
        model.wv.save(wv_file_path)
    
    try:
        wv = KeyedVectors.load(wv_file_path, mmap='r')
    except FileNotFoundError:
            print("Error while loading wv file: ", 
                  "wv file can't be found in ", 
                  wv_file_path, " file path!")
            sys.exit()
    
    return wv