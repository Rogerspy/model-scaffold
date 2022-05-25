#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bloom_embedding.py
@Time    :   2022/05/21 09:55:12
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import mmh3
import numpy as np
from gensim.models import KeyedVectors


class BloomEmbedding(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(
        self,
        nb_epoch: int=5000,
        learn_rate: float=0.001,
        nr_hash_vector: int=10000,
        nb_keys: int=3
    ) -> None:
        
        self.nb_epoch = nb_epoch
        self.learn_rate = learn_rate
        self.nr_hash_vector = nr_hash_vector
        self.nb_keys = nb_keys
        
        # 碰撞词
        self.collision = []
        # word keys
        self.word2keys = {}
        self.keys2word = {}
        self._hash_vectors = None
    
    def _get_word_key(self, word_emb):
        for word in word_emb.index_to_key:
            keys = tuple([mmh3.hash(word, seed=i) % self.nr_hash_vector for i in range(self.nb_keys)])
            self.word2keys[word] = keys
            if keys not in self.keys2word:
                self.keys2word[keys] = word
            else:
                self.collision.append(f'{word}-{self.keys2word[keys]}')
        print(f'一共有 {len(self.collision)} 个冲突词...')
    
    def train(self, pretrained_embedding) -> np.ndarray:
        word_embedding = KeyedVectors.load_word2vec_format(
            pretrained_embedding,
            binary=False, 
            encoding='utf-8', 
            unicode_errors='ignore'
        )
        
        if not self._hash_vectors:
            self._hash_vectors = np.random.uniform(
                -0.1, 0.1, (self.nr_hash_vector, word_embedding.vector_size)
            )
        
        self._get_word_key(word_embedding)
        
        min_loss = np.inf
        for epoch in range(self.nb_epoch):
            loss = 0.0
            for word in self.word2keys.keys():
                hash_vector = self.get_vector(word)
                diff = hash_vector - word_embedding[word]
                
                # 模拟梯度下降
                self._hash_vectors[list(self.word2keys[word])] -= self.learn_rate * diff
                loss += abs(diff).sum()
            print(f'==== epoch: {epoch} | loss: {loss: >0.3f} | diff: {diff.sum()} | abs_diff: {abs(diff).sum()} ====')
            if loss < min_loss:
                min_loss = loss
                np.save(f'hashed_vectors.npy', self._hash_vectors)
                
    def load_embedding(self, fpath):
        self._hash_vectors = np.load(fpath)
            
    def get_vector(self, word):
        keys = [mmh3.hash(word, seed=i) % self.nr_hash_vector for i in range(self.nb_keys)]
        hash_vector = sum([self._hash_vectors[k] for k in keys])
        return hash_vector
                
    def similar_by_word(self, word1, word2):
        """
        Computing similarity between `word1` and `word2` by cosine:
        sim = sum(A*B) / (np.sqrt(sum(A**2)) * np.sqrt(sum(B**2)))

        Args:
            word1 (str): _description_
            word2 (str): _description_

        Returns:
            float: similarity
        """
        vector1 = self.get_vector(word1)
        vector2 = self.get_vector(word2)

        num = sum(vector1 * vector2)   
        denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)  
        sim = num / denom
        return sim
