# -*- encoding: utf-8 -*-
'''
@File    :   pretrained_models.py
@Time    :   2024/07/09 22:42:23
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''

import os
import hashlib
import pickle as pkl
from tqdm import tqdm
from types import SimpleNamespace
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import MetadataMode

class EmbModel():
    def __init__(self,
                 model_name: str,
                 embed_batch_size: int = 512,
                 node_space: SimpleNamespace = None,
                 device: torch.device = torch.device('cuda:0'),
                 cache_root: str = "./cache"):
        self.model_name = model_name
        self.emb_model = HuggingFaceEmbedding(model_name=model_name,
                                              cache_folder="./",
                                              embed_batch_size=embed_batch_size,
                                              device=device)
        self.emb_dim = self.get_embedding_dim(model_name)
        self.embed_batch_size = embed_batch_size
        self.device = device
        self.cache_root = cache_root
        
        self.ids_dict, self.nodes_embeddings_dict = self.nodes2emb(node_space)
    
    @staticmethod
    def get_embedding_dim(model_name: str) -> int:
        if "bce-embedding-base_v1" in model_name:
            return 768
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    
    def nodes2emb(self, nodes_space):
        docs_names = list(nodes_space.__dict__.keys())
        if self.__cache_exist(docs_names):
            print("加载Embeddings缓存， 加载路径：", self.__cache_save_path)
            ids_dict = {}
            nodes_embeddings_dict = {}
            for name in tqdm(docs_names, desc="Loading embeddings from cache"):
                ids_dict[name], nodes_embeddings_dict[name] = self.__load_embeddings_cache(name)
        else:
            print("没有找到Embeddings缓存，重新构建Embeddings")
            ids_dict, text_chunks_dict = {}, {}
        
            for name in docs_names:
                nodes = getattr(nodes_space, name)
                ids_dict[name] = [node.id_ for node in nodes]
                text_chunks_dict[name] = [node.metadata["text_for_retrieval"] for node in nodes]

            nodes_embeddings_dict = {}
            for name in docs_names:
                print(f"Building embeddings for {name}...")
                nodes_embeddings_dict[name] = self.emb_model.get_text_embedding_batch(texts=text_chunks_dict[name], show_progress=True)
                self.__save_embeddings_cache(name, ids_dict[name], nodes_embeddings_dict[name])
        
        return ids_dict, nodes_embeddings_dict
    
    def get_text_embedding(self, text: str):
        return self.emb_model.get_text_embedding(text)
    
    @property
    def __cache_save_path(self) -> str:
        path = os.path.join(self.cache_root, "embeddings")
        return path
    
    def __cache_exist(self, doc_names)-> bool:
        return all([os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) for name in doc_names])
    
    def __save_embeddings_cache(self, name, ids, embeddings):
        save_path = os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))
        os.makedirs(self.__cache_save_path, exist_ok=True)
        with open(save_path, "wb") as f:
            pkl.dump((ids, embeddings), f)
        print(f"Embeddings cache saved to {save_path}")
    
    def __load_embeddings_cache(self, name):
        load_path = os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))
        with open(load_path, "rb") as f:
            return pkl.load(f)
        

        

class ReRank_Model:
    def __init__(self,
                 model_name: str,
                 device: torch.device = torch.device('cuda:0')):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
    
    def rerank(self, query:str, context: List[str]):
        with torch.no_grad():
            pairs = [[query, ctx] for ctx in context]
            # 输入长度不要超过1000
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            outputs = self.model(**inputs)
            scores = outputs.logits.view(-1).float().cpu()
        return scores