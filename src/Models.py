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
from FlagEmbedding import BGEM3FlagModel

class EmbModel():
    def __init__(self,
                 model_name: str,
                 embed_batch_size: int = 512,
                 node_space: SimpleNamespace = None,
                 device: torch.device = torch.device('cuda:0'),
                 cache_root: str = "./cache"):
        self.model_name = model_name
        if "m3" in self.model_name:
            self.emb_model = BGEM3FlagModel(model_name_or_path=model_name,  
                                            use_fp16=True, 
                                           device=device)
        else:
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
        elif "bge-m3" in model_name:
            return 1024
        elif "bge-large" in model_name:
            return 1024
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    
    def nodes2emb(self, nodes_space):
        docs_names = list(nodes_space.__dict__.keys())
        # print(docs_names)
        # exit()
        if self.__cache_exist(docs_names, any_exist=True):
            print("加载Embeddings缓存， 加载路径：", self.__cache_save_path)
            cache_exist_list, cache_notexist_list = self.__cache_exist_which(docs_names)
            print("已处理好的Embeddings缓存为：", cache_exist_list)
            print("未处理的Embeddings缓存为：", cache_notexist_list)
            ids_dict = {}
            nodes_embeddings_dict = {}
            for name in tqdm(docs_names, desc="Loading embeddings from cache"):
                if name not in cache_exist_list:
                    ids_dict[name], nodes_embeddings_dict[name] = self.node2emb_process(nodes_space, name)
                    continue
                ids_dict[name], nodes_embeddings_dict[name] = self.__load_embeddings_cache(name)
        else:
            print("没有找到Embeddings缓存，重新构建Embeddings")
            ids_dict, text_chunks_dict = {}, {}

            nodes_embeddings_dict = {}
            for name in docs_names:
                ids_dict[name], nodes_embeddings_dict[name] = self.node2emb_process(nodes_space, name)
        #         nodes = getattr(nodes_space, name)
        #         ids_dict[name] = [node.id_ for node in nodes]
        #         text_chunks_dict[name] = [node.metadata["text_for_retrieval"] for node in nodes]
        #         print(f"Building embeddings for {name}...")
        #         nodes_embeddings_dict[name] = self.emb_model.get_text_embedding_batch(texts=text_chunks_dict[name], show_progress=True)
        #         self.__save_embeddings_cache(name, ids_dict[name], nodes_embeddings_dict[name])

        #     # nodes_embeddings_dict = {}
            # for name in docs_names:
            #     print(f"Building embeddings for {name}...")
            #     nodes_embeddings_dict[name] = self.emb_model.get_text_embedding_batch(texts=text_chunks_dict[name], show_progress=True)
            #     self.__save_embeddings_cache(name, ids_dict[name], nodes_embeddings_dict[name])
        
        return ids_dict, nodes_embeddings_dict

    def node2emb_process(self, nodes_space, doc_name):
        nodes = getattr(nodes_space, doc_name)
        ids_dict = [node.id_ for node in nodes]
        text_chunks_dict = [node.metadata["text_for_retrieval"] for node in nodes]
        print(f"Building embeddings for {doc_name}...")
        if "m3" in self.model_name:
            nodes_embeddings_dict = self.emb_model.encode(sentences=text_chunks_dict, batch_size=self.embed_batch_size, return_dense=True, return_sparse=True, return_colbert_vecs=True)['colbert_vecs']
        else:
            nodes_embeddings_dict = self.emb_model.get_text_embedding_batch(texts=text_chunks_dict, show_progress=True)
        self.__save_embeddings_cache(doc_name, ids_dict, nodes_embeddings_dict)
        return ids_dict, nodes_embeddings_dict
        
    
    def get_text_embedding(self, text: str):
        return self.emb_model.get_text_embedding(text)
    
    @property
    def __cache_save_path(self) -> str:
        path = os.path.join(self.cache_root, "embeddings")
        return path
    
    def __cache_exist(self, doc_names, any_exist=False)-> bool:
        if any_exist:
            return any([os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) for name in doc_names])
        return all([os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) for name in doc_names])

    def __cache_exist_which(self, docs_names)-> list:
        # return all([os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) for name in doc_names])
        # return [os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) if os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) for name in doc_names]
        exist_list, not_exist_list = [], []
        for name in docs_names:
            if os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))):
                exist_list.append(name)
            else:
                not_exist_list.append(name)
        return exist_list, not_exist_list
        # return [name if os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) else print() for name in doc_names] [name if not os.path.exists(os.path.join(self.__cache_save_path, "{}_embeddings.pkl".format(name))) else print() for name in doc_names]
    
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