# -*- encoding: utf-8 -*-
'''
@File    :   qa.py
@Time    :   2024/07/09 23:50:22
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''

from typing import List
import jsonlines
import json
import numpy as np
import pandas as pd

from llama_index.legacy.llms import OpenAILike
from llama_index.core import PromptTemplate

import src.Config as config

def load_query(path: str):
    print(f"Loading queries from {path}")
    queries = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            queries.append(obj)
    return queries

def load_query_source(path:str):
    print(f"Loading queries source from {path}")
    with open(path, "r") as f:
        id2source = json.load(f)
    return id2source

def build_prompt(query:str, retrievals:List, max_retrieval_num:int):
    str_list = ["文档路径：" + res.metadata["file_path"] + "\n" + 
                res.metadata["text_for_rerank_and_answer"] for res in retrievals]
    str_list = str_list[:max_retrieval_num]
    
    context_str = "\n----------文档分隔符---------\n".join(str_list)
    prompt = PromptTemplate(config.QA_TEMPLATE).format(context_str=context_str, query_str=query)
    return prompt
    

class QA:
    def __init__(self,
                 queries: List[dict],
                 retrievals: dict[dict],
                 id2source: dict = None):
        self.query_ids = [q["id"] for q in queries]
        self.query_str = [q["query"] for q in queries]
        self.query_doc = [q["document"] for q in queries]
        self.query_ref = [q["answer"] if "answer" in q else None for q in queries] # 参考答案
        self.query_src = [q["source"] if "source" in q else None for q in queries] # 来源
        self.queryid2source = id2source
        
        self.retrievals_combined = [rr["rr_res"] for k, rr in retrievals.items()]
        self.retrievals_dense = [rr["emb_res"] for k, rr in retrievals.items()]
        self.retrievals_sparse = [rr["bm25_res"] for k, rr in retrievals.items()]
        
        self.query_ans = [None for q in queries] # 待生成的答案
        
        self.config = config
        
        self.LLM = self.load_llm()
    
    def load_llm(self):
        return OpenAILike(api_key=self.config.GLM_KEY,
                          model="glm-4",
                          api_base="https://open.bigmodel.cn/api/paas/v4/",
                          is_chat_model=True)
        
    def refer_answers(self, id):
        idx = id - 1
        return self.query_ref[idx]
        
    def answer(self, id:int, max_retrieval_num:int):
        # id 是 query 的 id
        # max_retrieval_num 是最多使用的检索结果数量
        idx = id - 1
        query = self.query_str[idx]
        retrievals = self.retrievals_combined[idx] 
        prompt = build_prompt(query, retrievals, max_retrieval_num)
        ret = self.LLM.complete(prompt)
        return id, prompt, ret
    

    def retrieval_performence(self, cache_path:str):
        # 召回结果中，真实文档的排名
        max_retrieval_num = 20
        
        ids_used = [int(k) for k, v in self.queryid2source.items()]
        combined_rank, dense_rank, sparse_rank = [], [], []
        combined_hits_at_5, dense_hits_at_5, sparse_hits_at_5 = 0, 0, 0
        combined_hits_at_10, dense_hits_at_10, sparse_hits_at_10 = 0, 0, 0
        combined_hits_at_15, dense_hits_at_15, sparse_hits_at_15 = 0, 0, 0
        combined_hits_at_30, dense_hits_at_30, sparse_hits_at_30 = 0, 0, 0
        
        for id in ids_used:
            source_path = self.queryid2source[str(id)]
            combined_paths = [res.metadata["file_path"] for res in self.retrievals_combined[id-1]][:max_retrieval_num]
            dense_paths = [res.metadata["file_path"] for res in self.retrievals_dense[id-1]][:max_retrieval_num]
            sparse_paths = [res.metadata["file_path"] for res in self.retrievals_sparse[id-1]][:max_retrieval_num]
            
            # 计算排名
            if source_path in combined_paths:
                combined_rank.append(combined_paths.index(source_path) + 1)
            else:
                combined_rank.append(1e+4)
            
            if source_path in dense_paths:
                dense_rank.append(dense_paths.index(source_path) + 1)
            else:
                dense_rank.append(1e+4)
            
            if source_path in sparse_paths:
                sparse_rank.append(sparse_paths.index(source_path) + 1)
            else:
                sparse_rank.append(1e+4)
            
            # 计算命中@5
            if source_path in combined_paths[:5]:
                combined_hits_at_5 += 1
            if source_path in dense_paths[:5]:
                dense_hits_at_5 += 1
            if source_path in sparse_paths[:5]:
                sparse_hits_at_5 += 1
            
            # 计算命中@10
            if source_path in combined_paths[:10]:
                combined_hits_at_10 += 1
            if source_path in dense_paths[:10]:
                dense_hits_at_10 += 1
            if source_path in sparse_paths[:10]:
                sparse_hits_at_10 += 1
                
            # 计算命中@15
            if source_path in combined_paths[:15]:
                combined_hits_at_15 += 1
            if source_path in dense_paths[:15]:
                dense_hits_at_15 += 1
            if source_path in sparse_paths[:15]:
                sparse_hits_at_15 += 1
            
            # 计算命中@30  
            if source_path in combined_paths[:30]:
                combined_hits_at_30 += 1
            if source_path in dense_paths[:30]:
                dense_hits_at_30 += 1
            if source_path in sparse_paths[:30]:
                sparse_hits_at_30 += 1
        
        # 计算倒数的平均值
        combined_mrr = np.mean(1/np.array(combined_rank))
        dense_mrr = np.mean(1/np.array(dense_rank))
        sparse_mrr = np.mean(1/np.array(sparse_rank))
        
        # 计算精度@5 和 @10
        num_queries = len(ids_used)
        combined_precision_at_5 = combined_hits_at_5 / num_queries
        dense_precision_at_5 = dense_hits_at_5 / num_queries
        sparse_precision_at_5 = sparse_hits_at_5 / num_queries
        
        combined_precision_at_10 = combined_hits_at_10 / num_queries
        dense_precision_at_10 = dense_hits_at_10 / num_queries
        sparse_precision_at_10 = sparse_hits_at_10 / num_queries
        
        combined_precision_at_15 = combined_hits_at_15 / num_queries
        dense_precision_at_15 = dense_hits_at_15 / num_queries
        sparse_precision_at_15 = sparse_hits_at_15 / num_queries
        
        sparse_hits_at_30 = sparse_hits_at_30 / num_queries
        dense_hits_at_30 = dense_hits_at_30 / num_queries
        combined_hits_at_30 = combined_hits_at_30 / num_queries
        
        # 打印结果
        print("\nRetrieval Performance Metrics\n" + "="*30)
        print(f"{'Metric':<20} {'Combined':<10} {'Dense':<10} {'Sparse':<10}")
        print(f"{'MRR':<20} {combined_mrr:<10.4f} {dense_mrr:<10.4f} {sparse_mrr:<10.4f}")
        print(f"{'Precision @5':<20} {combined_precision_at_5:<10.4f} {dense_precision_at_5:<10.4f} {sparse_precision_at_5:<10.4f}")
        print(f"{'Precision @10':<20} {combined_precision_at_10:<10.4f} {dense_precision_at_10:<10.4f} {sparse_precision_at_10:<10.4f}")
        print(f"{'Precision @15':<20} {combined_precision_at_15:<10.4f} {dense_precision_at_15:<10.4f} {sparse_precision_at_15:<10.4f}")
        print(f"{'Precision @30':<20} {combined_hits_at_30:<10.4f} {dense_hits_at_30:<10.4f} {sparse_hits_at_30:<10.4f}")
        
        df = pd.DataFrame({
            "Metric": ["MRR", "Precision @5", "Precision @10", "Precision @15", "Precision @30"],
            "Combined": [combined_mrr, combined_precision_at_5, combined_precision_at_10, combined_precision_at_15, combined_hits_at_30],
            "Dense": [dense_mrr, dense_precision_at_5, dense_precision_at_10, dense_precision_at_15, dense_hits_at_30],
            "Sparse": [sparse_mrr, sparse_precision_at_5, sparse_precision_at_10, sparse_precision_at_15, sparse_hits_at_30]})
        
        df.to_csv(cache_path + "/retrieval_performance.csv", index=False)
            
        
        
            
            
            
        
        
        

# def load_llm(config):
#     llm = OpenAILike(api_key=config.GLM_KEY,
#                      model="glm-4",
#                      api_base="https://open.bigmodel.cn/api/paas/v4/",
#                      is_chat_model=True)
#     return llm

# def generate_answer(llm : OpenAILike, query:str, retrievals: List, TEMPLATE:str, max_str_len: int, tag:int):
#     context_str = ""
#     for res in retrievals:
#         if info_len(context_str + res.metadata["window"]) <= max_str_len:
#             context_str += res.metadata["window"] + "\n==========文档分割符============\n"
#         else:
#             break
#     prompt = PromptTemplate(TEMPLATE).format(context_str=context_str,
#                                              query_str=query)
#     ret = llm.complete(prompt)
#     return tag, ret

# def enhance_query(llm: OpenAILike, query: str, doc_name: str, retriever, config, bm25_topk = 10, emb_topk = 10, max_str_len = 25000):
#     # 检索
#     retriever_results, _, _ = retriever.retrieval(query = query,
#                                                   doc_name = doc_name,
#                                                   bm25_topk = bm25_topk,
#                                                   emb_topk = emb_topk)
#     # 重排
#     rerank_results, _ = retriever.rerank(query, retriever_results)

#     _, answer = generate_answer(llm = llm,
#                                 query = query,
#                                 retrievals = rerank_results,
#                                 TEMPLATE = config.RW_TEMPLATE,
#                                 max_str_len = max_str_len,
#                                 tag = 0)
#     enhanced_query = answer
    
#     return enhanced_query, retriever_results, rerank_results

# def retrieve_and_answer(llm: OpenAILike,
#                         query: str,
#                         enhanced_query: str,
#                         doc_name: str,
#                         retriever,
#                         config,
#                         bm25_topk = 50,
#                         emb_topk = 50,
#                         max_str_len = 25000):
#     # 检索
#     retriever_results, _, _ = retriever.retrieval(query = query + "\n" + enhanced_query,
#                                                   doc_name = doc_name,
#                                                   bm25_topk = bm25_topk,
#                                                   emb_topk = emb_topk)
#     # 重排
#     rerank_results, _ = retriever.rerank(query + "\n" + enhanced_query,
#                                          retriever_results)

#     _, answer = generate_answer(llm = llm,
#                                 query = query,
#                                 retrievals = rerank_results,
#                                 TEMPLATE = config.QA_TEMPLATE,
#                                 max_str_len = max_str_len,
#                                 tag = -1)
#     answer = answer
    
#     return answer, retriever_results, rerank_results


# def enhance_query_batch_api(queries, config, retriever, bm25_topk, emb_topk, max_str_len):
#     retriever_docs = []
#     for query_dict in tqdm(queries, total=len(queries), desc="利用初始query检索文档"):
#         query = query_dict["query"]
#         doc_name = query_dict["document"]
#         retriever_results, _, _ = retriever.retrieval(query = query,
#                                                       doc_name = doc_name,
#                                                       bm25_topk = bm25_topk,
#                                                       emb_topk = emb_topk)
#         rerank_results, _ = retriever.rerank(query, retriever_results)
#         retriever_docs.append(rerank_results)
    
#     # 构建上下文
#     context_strs = []
#     for retrievals in retriever_docs:
#         context = ""
#         for res in retrievals:
#             if info_len(context + res.metadata["window"]) <= max_str_len:
#                 context += res.metadata["window"] + "\n==========文档分割符============\n"
#             else:
#                 break
#         context_strs.append(context)
    
#     # 构建prompt
#     prompts = []
#     for query_dict, context_str in zip(queries, context_strs):
#         query = query_dict["query"]
#         prompt = PromptTemplate(config.RW_TEMPLATE).format(context_str=context_str,
#                                                            query_str=query)
#         prompts.append(prompt)
    
#     # 批量api调用文件
#     batch_api_requests = []
#     for id, p in enumerate(prompts):
#         tmp = config.BATCH_TEMPLATE.copy()
#         tmp["custom_id"] = f"query_rw_{id}"
#         tmp["body"]["messages"][1]["content"] = p
#         batch_api_requests.append(tmp)
    
#     # 保存到本地jsonl文件
#     batch_api_file_name = "batch_api_requests_rewrite.jsonl"
#     with open(batch_api_file_name, "w") as f:
#         for req in batch_api_requests:
#             f.write(jsonlines.dumps(req))
    
   

#     client = ZhipuAI(api_key=config.GLM_KEY)
    
#     result = client.files.create(
#         file=open(batch_api_file_name, "rb"),
#         purpose="batch")
    
#     create = client.batches.create(
#         input_file_id=result.id,
#         endpoint="/v4/chat/completions", 
#         completion_window="24h", #完成时间只支持 24 小时
#         metadata={"description": "batch 增强"})
    
    