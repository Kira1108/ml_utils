import os
import sys

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import random
from typing import Dict, Union
from abc import ABC, abstractmethod


@dataclass
class Seq2SeqDataset:

    file_path:str
    num_samples:int

    def __post_init__(self):
        self.fitted = False

    def _create(self):
        self.input_texts = []
        self.target_texts = []
        self.target_texts_inputs = []


    def create_dataset(self):
        self._create()
        t = 0

        for line in open(self.file_path):

            t += 1
            if t > self.num_samples:
                break

            if '\t' not in line:
                continue
            
            input_text,  translation, _ = line.split("\t")
            target_text = translation + " <eos>"
            target_text_input = "<sos> " + translation

            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            self.target_texts_inputs.append(target_text_input)

        self.fitted = True

        return self.input_texts, self.target_texts, self.target_texts_inputs

    def load(self):
        if self.fitted:
            return self.input_texts, self.target_texts, self.target_texts_inputs

        else:
            return self.create_dataset()

    def inspect(self, n = 10):
        for _ in range(n):
            idx = random.randint(0, len(self.input_texts) - 1)
            print("*"*20)
            print(f"Input: {self.input_texts[idx]}")
            print(f"Target: {self.target_texts[idx]}")
            print(f"Target_inputs: {self.target_texts_inputs[idx]}")
            print("*"*20)

class MyModels(ABC):
    def fit(self, *args, **kwargs):
        return self

    @abstractmethod
    def transform(self, *args, **kwargs):
        """Perform transformation"""
        ...

    def fit_transform(self, *args, **kwargs):
        if (not hasattr(self,'fitted')) or (not self.fitted):
            self.fit(*args, **kwargs)
            self.fitted = True
        return self.transform(*args,**kwargs)


    def __call__(self, *args, **kwargs):
        if (not hasattr(self,'fitted')) or (not self.fitted):
            self.fit()

        return self.transform(*args, **kwargs)

class SeqTokenizer(MyModels):
    
    def __init__(self, max_num_words, *args, **kwargs):
        self.max_num_words = max_num_words
        self.tokenizer = Tokenizer(num_words = self.max_num_words, 
                                   *args, 
                                   **kwargs)
        self.fitted = False

    def fit(self, inputs):
        self.tokenizer.fit_on_texts(inputs)
        self.fitted = True
        return self

    def transform(self, inputs):
        return self.tokenizer.texts_to_sequences(inputs)

    def inverse_list(self, idx_inputs:list):
        return [self.tokenizer.index_word.get(idx,"<UNK>") for idx in idx_inputs]

    def inverse_batch(self, idx_inputs:list):
        return [self.inverse_list(l) for l in idx_inputs]

    @property
    def word_index(self):
        return self.tokenizer.word_index

    @property
    def index_word(self):
        return self.tokenizer.index_word

@dataclass
class TokenizerContainer(MyModels):

    max_num_words:int = 10000
    max_num_words_inputs:int = None
    max_num_words_outputs:int = None

    def __post_init__(self):

        if not self.max_num_words_inputs:
            self.max_num_words_inputs = self.max_num_words

        if not self.max_num_words_outputs:
            self.max_num_words_outputs = self.max_num_words

        self.tokenizer_inputs = SeqTokenizer(self.max_num_words_inputs)
        self.tokenizer_outputs = SeqTokenizer(self.max_num_words_outputs, filters = '')
        self.fitted = False

    def fit(self,
            input_texts:Union[list, np.ndarray], 
            target_texts:Union[list, np.ndarray],
            target_texts_inputs:Union[list, np.ndarray]):
        
        self.tokenizer_inputs.fit(input_texts)
        self.tokenizer_outputs.fit(target_texts + target_texts_inputs)
        self.fitted = True
        return self

    def transform(self,
            input_texts:Union[list, np.ndarray], 
            target_texts:Union[list, np.ndarray],
            target_texts_inputs:Union[list, np.ndarray]):
        
        input_sequences = self.tokenizer_inputs.transform(input_texts)
        target_sequences = self.tokenizer_outputs.transform(target_texts)
        target_sequences_inputs = self.tokenizer_outputs.transform(target_texts_inputs)
        
        return input_sequences,target_sequences,target_sequences_inputs

@dataclass
class SequenceStats:
    input_sequences:list
    target_sequences:list
    target_sequences_inputs:list
    tokenizer_container: TokenizerContainer

    @staticmethod
    def get_max_length(inputs):
        return max([len(l) for l in inputs])

    @property
    def max_len_inputs(self):
        return self.get_max_length(self.input_sequences)

    @property
    def max_len_targets(self):
        return self.get_max_length(self.target_sequences)

    @property
    def max_len_target_inputs(self):
        return self.get_max_length(self.target_sequences_inputs)

    @property
    def num_words_inputs(self):
        return len(self.tokenizer_container.tokenizer_inputs.word_index) + 1

    @property
    def num_words_targets(self):
        return len(self.tokenizer_container.tokenizer_outputs.word_index) + 1

    def to_dict(self):
        return dict(
            max_len_inputs =  self.max_len_inputs,
            max_len_targets = self.max_len_targets,
            max_len_target_inputs = self.max_len_target_inputs,
            num_words_inputs = self.num_words_inputs,
            num_words_targets = self.num_words_targets
        )

def Padder(stats:SequenceStats):
    def pad(input_sequences:list, 
            target_sequences:list, 
            target_sequences_inputs:list):
        
        encoder_inputs = pad_sequences(input_sequences, 
                                    maxlen = stats.max_len_inputs)
        
        decoder_inputs = pad_sequences(target_sequences_inputs, 
                                    maxlen = stats.max_len_targets, 
                                    padding = 'post')
        
        decoder_targets = pad_sequences(target_sequences, 
                                        maxlen = stats.max_len_targets, 
                                        padding = 'post')
        
        return encoder_inputs, decoder_inputs, decoder_targets
    return pad


@dataclass
class EmbeddingCreator:

    embedding_path:str
    word_index:dict
    embedding_dim:int
    max_num_words:int = None

    def _load_embedding_vectors(self):
        word2vec = {}

        with open(self.embedding_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.array(values[1:], dtype = np.float32)
                word2vec[word] = vec
            return word2vec

    def _fill_embedding_matrix(self, word2vec):
        
        words_dataset = len(self.word_index) + 1

        if not self.max_num_words:
            self.max_num_words = words_dataset

        num_words = min(self.max_num_words, words_dataset)
        self.num_words = num_words
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        for word, i, in self.word_index.items():
            if i < self.max_num_words:
                embedding_vector = word2vec.get(word, None)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def create_embedding_matrix(self):
        word2vec = self._load_embedding_vectors()
        embedding_matrix = self._fill_embedding_matrix(word2vec)
        return embedding_matrix

    def __call__(self, **kwargs):
        matrix = self.create_embedding_matrix()
        return Embedding(self.num_words, self.embedding_dim, weights = [matrix], **kwargs)

def create_onehot_sequence(sequences, V, N = None, T = None):
    if not N:
        N = sequences.shape[0]
    if not T:
        T = sequences.shape[1]
    
    onehot = np.zeros((N ,T, V), dtype = np.float32)

    # nn-batch dimension
    # tt-time dimension
    # vv-word index dimension
    for nn, sequence in enumerate(sequences):
        for tt, vv in enumerate(sequence):
            onehot[nn, tt, vv] = 1

    return onehot