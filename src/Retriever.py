# -*- encoding: utf-8 -*-
'''
@File    :   retriever.py
@Time    :   2024/07/09 23:26:37
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''

import os
import re
import copy
import hashlib
import pickle as pkl
from typing import List
from tqdm import tqdm
from collections import defaultdict

from src.TextPrcess import Docs_Nodes
from src.Models import EmbModel, ReRank_Model
from src.BM25 import BM25
from src.VectorStore import VectorStore

import torch


class Retriever:
    def __init__(self,
                 docs_nodes: Docs_Nodes,
                 emb_model: EmbModel,
                 rr_model: ReRank_Model,
                 bm25: BM25,
                 vector_store: VectorStore,
                 device: torch.device,
                 cache_root: str = "./cache"):
        self.docs_nodes = docs_nodes
        self.nodes_space = docs_nodes.nodes_space
        self.emb_model = emb_model
        self.rr_model = rr_model
        self.bm25 = bm25
        self.vector_store = vector_store
        self.device = device
        self.cache_root = cache_root 
        
    
    def query2dense(self, query: str):
        return self.emb_model.get_text_embedding(query)
    
    def query2sparse(self, query: str):
        return self.bm25.cut_words(query)
        
    def bm25_retrieval(self, query: str, doc_name: str, k: int):
        nodes = getattr(self.nodes_space, doc_name)
        bm25  = getattr(self.bm25.BM25_space, doc_name)
        res = bm25.get_top_n(self.query2sparse(query), nodes, n=k)
        return res 

    def emb_retrieval(self, query: str, doc_name: str, k: int):
        collection_name = doc_name + "_collection"
        client = self.vector_store.qdrant_client
        
        # 执行向量检索
        res = client.search(
            collection_name=collection_name,
            query_vector=self.query2dense(query),
            query_filter=None,
            limit=k)
        
        # 将 Qdrant 检索结果转换为文档对象
        nodes = getattr(self.nodes_space, doc_name)
        id2doc = {doc.id_: doc for doc in nodes}
        res = [id2doc[r.id] for r in res]
        
        return res

    def retrieval(self, query: str, doc_name: str, bm25_topk: int = 50, emb_topk: int = 50):
        # BM25检索
        bm25_res = self.bm25_retrieval(query, doc_name, bm25_topk)
        # 嵌入检索
        emb_res = self.emb_retrieval(query, doc_name, emb_topk)
        
        search_res = [] # 合并后的检索结果
        search_ids = [] # 用于去重
        search_str = [] # 用于去重
        
        for item in bm25_res + emb_res:
            if item.id_ not in search_ids:
                pure_str = "".join(re.findall(r"\w", item.metadata["text_for_answer"]))
                if all([(s not in pure_str) and (pure_str not in s) for s in search_str]):    
                    search_res.append(copy.deepcopy(item)) # 注意，一定要深拷贝
                    search_ids.append(item.id_)
                    search_str.append(pure_str)
        
        return search_res, bm25_res, emb_res
    
    def rerank(self, query: str, retrievals: List):
        context = [doc.metadata["text_for_rerank"] for doc in retrievals]
        rr_scores = self.rr_model.rerank(query, context)
        rr_res = [doc for _, doc in sorted(zip(rr_scores, retrievals),
                                           key=lambda x:x[0],
                                           reverse=True)]
        return rr_res, rr_scores
    
    def batch_retrieval_and_rerank(self, 
                                   debug_queries: List[dict],
                                   submit_queries: List[dict],
                                   bm25_topk: int,
                                   emb_topk: int):
        for tag, queries in [("debug", debug_queries), ("submit", submit_queries)]:
            results = defaultdict(dict)
            for query_dict in tqdm(queries, total=len(queries), desc="Batch retrieval and rerank"):
                query = query_dict["query"]
                doc_name = query_dict["document"]
                all_res, bm25_res, emb_res = self.retrieval(query = query,
                                                            doc_name = doc_name,
                                                            bm25_topk = bm25_topk,
                                                            emb_topk = emb_topk)

                rr_res, rr_score = self.rerank(query, all_res)
                results[query_dict["id"]] = {
                    "query": query,
                    "document": doc_name,
                    "all_res": all_res,
                    "bm25_res": bm25_res,
                    "emb_res": emb_res,
                    "rr_res": rr_res,
                    "rr_score": rr_score
                }
                
            self.__save_cache(results, tag)
    
    def __save_cache(self, results: dict, tag: str):
        save_path = os.path.join(self.cache_root, "retrieval_rerank_results_{}.pkl".format(tag))
        with open(save_path, "wb") as f:
            pkl.dump(results, f)
        print("Results saved to: ", save_path)
        