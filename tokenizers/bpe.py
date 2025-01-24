#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bpe.py
@Time    :   2023/09/28 11:16:17
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import os
import re
import collections

from cortex.utils import progress
from cortex.utils import Log


class BPETokenizer(object):
    def __init__(self, log_path: str | None = None, tokenizer_path: str | None = None) -> None:
        self.tokenizer_path = tokenizer_path
        self.logger = Log(log_file_path=log_path)
        self.vocabulary = []
        self.merges = collections.OrderedDict()

        self.punctuation = re.compile(r'([^\w\s]+|\d+[,.\s\d]*\d*)')

    @staticmethod
    def _get_token_length(token):
        return len(token)

    @staticmethod
    def preprocess(text: str):
        p = re.compile(r'([a-zA-Z]+)')
        text = p.sub(r' \1 ', text)
        return text

    def get_vocab(self, file_path: str) -> dict:
        vocab = collections.defaultdict(int)
        with open(file_path, encoding='utf8') as f:
            lines = f.readlines()
            for line in progress(lines, f'BPE getting vocabulary from {file_path}'):
                line = self.preprocess(line)
                words = line.strip().split()
                for word in words:
                    vocab[' '.join(list(word)) + ' </w>'] += 1
        self.logger.info(f'Build BPE vocabulary is finished with {len(vocab.keys())} unique words.')
        return vocab

    def get_statistics(self, vocab: dict) -> dict:
        pairs = collections.defaultdict(int)

        for word, freq in progress(vocab.items(), 'Stats vocab'): 
            symbols =  word.split()
            for idx in range(len(symbols)-1):
                pairs[symbols[idx], symbols[idx+1]] += freq
        # self.logger.info(f'Statistics symbol pairs is finished with {len(pairs.keys())}.')
        return pairs

    def merge_vocab(self, pair, input_vocab):
        output_vocab = {}
        bigram = re.escape(' '.join(pair))
        print(bigram)
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in input_vocab:
            w_out = p.sub(''.join(pair), word)
            output_vocab[w_out] = input_vocab[word]
        
        return output_vocab

    def get_tokens_from_vocab(self, vocab: dict):
        tokens_freq = collections.defaultdict(int)
        for word, freq in vocab.items():
            tokens = word.split()
            for token in tokens:
                tokens_freq[token] += freq
        return tokens_freq

    def save_vocab_and_merges(self, vocab: list, merges: dict, save_dir: str):
        save_vocab_path = os.path.join(save_dir, 'vocab.txt')
        save_merges_path = os.path.join(save_dir, 'merges.txt')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # ordered_dict = collections.OrderedDict()
        # vocab = []
        # token_freq_sorted = sorted(token_freq.items(), key=lambda x: x[-1], reverse=True)
        # for k, _ in token_freq_sorted:
        #     vocab.append(k)

        with open(save_vocab_path, 'w', encoding='utf8') as f:
            f.write('\n'.join(vocab))

        with open(save_merges_path, 'w', encoding='utf8') as f:
            for line in merges.keys():
                f.write(' '.join(line)+'\n')
        self.logger.info(f'Save vocabulary to {save_vocab_path}.')

    def load_vocab_and_merges(self, load_dir: str):
        save_vocab_path = os.path.join(load_dir, 'vocab.txt')
        save_merges_path = os.path.join(load_dir, 'merges.txt')
        if not os.path.exists(save_vocab_path):
            self.logger.error(f'{save_vocab_path} is not exists, please check it again!')
        if not os.path.exists(save_merges_path):
            self.logger.error(f'{save_merges_path} is not exists, please check it again!')
        
        with open(save_vocab_path, encoding='utf8') as f:
            self.vocabulary = [x.strip() for x in f.readlines()]

        with open(save_merges_path, encoding='utf8') as f:
            for line in f:
                line = tuple(line.strip().split())
                self.merges[line] = ''.join(line)
        self.logger.info('Vocabulary and merges load success.')

    def _tokenize(self, sentence: str):
        if sentence == '':
            return ['']
        sort_merge = sorted(self.merges.items(), key=lambda x: len(x[1]), reverse=True)
        merges =collections.OrderedDict()
        for k, v in sort_merge:
            merges[k] = v
        sentence = self.punctuation.sub(r' \1 ', sentence)
        words = sentence.split()
        char_list = [[char for char in word] for word in words]
        for pair, merge in merges.items():
            for idx, chars in enumerate(char_list):
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == pair[0] and chars[i+1] == pair[1]:
                        chars = chars[:i] + [merge] + chars[i+2:]
                    else:
                        i += 1
                char_list[idx] = chars
        return sum(char_list, [])

    def tokenize(self, sequences: list):
        tokens_list = []
        for seq in sequences:
            tokens = self._tokenize(seq)
            tokens_list.append(tokens)
        return tokens_list

    def _detokenize(self, tokens: list):

        pass

    def detokenize(self, tokens_list: list):

        pass

    def _convert_tokens_to_ids(self):
        pass

    def convert_tokens_to_ids(self):
        pass

    def _convert_ids_to_tokens(self):
        pass

    def convert_ids_to_tokens(self):
        pass

    def train(self, file_path, num_merges: int = 10, min_freq: int = 1, vocab_size: int | None = None):
        vocab = self.get_vocab(file_path)
        for idx in progress(range(num_merges), 'Training tokenizer'):
            pairs = self.get_statistics(vocab)
            if not pairs:
                break
            max_freq_pair = max(pairs, key=pairs.get) # type: ignore
            self.merges[max_freq_pair] = ''.join(max_freq_pair)
            if pairs[max_freq_pair] <= min_freq:
                break
            print(max_freq_pair)
            vocab = self.merge_vocab(max_freq_pair, vocab)
            if vocab_size and len(vocab.keys()) <= vocab_size:
                break
            self.logger.info(f'Iter: {idx} | max_freq_pair: {max_freq_pair} - {pairs[max_freq_pair]}')
        self.vocabulary = list(self.get_tokens_from_vocab(vocab).keys())
        self.logger.info(f'Training is done. Number of tokens: {len(self.vocabulary)}')
