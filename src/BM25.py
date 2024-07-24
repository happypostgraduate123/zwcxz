# -*- encoding: utf-8 -*-
'''
@File    :   BM25.py
@Time    :   2024/07/09 23:17:32
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''

import os
import jieba
import hashlib
import pickle as pkl
from collections import defaultdict
from typing import List
from rank_bm25 import BM25Okapi
from llama_index.core.schema import MetadataMode

from tqdm import tqdm
from types import SimpleNamespace

class BM25:
    def __init__(self,
                 nodes_space: SimpleNamespace,
                 keywords_file_path: str,
                 cut_for_search: bool = False,
                 cache_root: str = "./cache"):
        self.nodes_space = nodes_space
        self.keywords_file_path = keywords_file_path
        self.keywords: List[str] = None
        self.cut_for_search = cut_for_search
        self.cache_root = cache_root
        self.keywords = self.load_keywords()
        
        self.BM25_space = self.build_BM25(self.nodes_space)
    
    def load_keywords(self):
        # 初始化 Jieba 并加载默认词典
        jieba.initialize()
        with open(self.keywords_file_path, "r", encoding="utf-8") as f:
            keywords = [w.strip() for w in f.readlines()]
        keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        for keyword in keywords:
            jieba.add_word(keyword)
        print("Added {} keywords to Jieba.".format(len(keywords)))
        return keywords
    
    def cut_words(self, doc: str) -> List[str]:
        if self.cut_for_search:
            return jieba.lcut_for_search(doc)
        else:
            return jieba.lcut(doc)

    def build_BM25(self, nodes_space: SimpleNamespace) -> SimpleNamespace:
        doc_names = list(nodes_space.__dict__.keys())
        # 提取需要分词的文本
        text_dict = defaultdict(list)
        for name in doc_names:
            nodes = getattr(nodes_space, name)
            for node in nodes:
                text_dict[name].append(node.metadata["text_for_retrieval"])
        
        if self.__cache_exit():
            print("Loading BM25 from cache path {}".format(self.__cache_path))
            bm25_space = self.__load_cache()
        else:
            print("No cache found, building BM25...")
            bm25_space = SimpleNamespace()
            for name in doc_names:
                text_list = text_dict[name]
                tokenized_corpus = []
                for text in tqdm(text_list, total=len(text_list), desc="Jieba cutting {:^8}".format(name)):
                    tokenized_corpus.append(self.cut_words(text))
                setattr(bm25_space, name, BM25Okapi(tokenized_corpus))
            self.__save_cache(bm25_space)
        
        return bm25_space
    
    def __cache_exit(self) -> bool:
        return os.path.exists(self.__cache_path)
    
    @property
    def __cache_path(self)-> str:
        return os.path.join(self.cache_root, "bm25.pkl")
    
    def __load_cache(self) -> SimpleNamespace:
        with open(self.__cache_path, "rb") as f:
            return pkl.load(f)
    
    def __save_cache(self, bm25_space: SimpleNamespace):
        with open(self.__cache_path, "wb") as f:
            pkl.dump(bm25_space, f)
        print("BM25 cache saved at {}".format(self.__cache_path))