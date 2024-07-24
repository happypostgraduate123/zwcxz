# -*- encoding: utf-8 -*-
'''
@File    :   build_context.py
@Time    :   2024/07/20 00:29:51
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''
import os
import time
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import nest_asyncio
nest_asyncio.apply()

from src.TextPrcess import Docs_Nodes
from src.Models import EmbModel, ReRank_Model
from src.VectorStore import VectorStore
from src.BM25 import BM25
from src.Retriever import Retriever
from src.QA import load_query
from src.Utils import infer_dvice

def build_cache(args):
    # require the custom exps name for this run
    if not args.exp_tag:
        print("输入本次实验的自定义标签：", end="")
        args.exp_tag = input()
    time_now = time.strftime("%Y-%m-%d", time.localtime())
    args.cache_root = f"{args.cache_root}/{time_now}-exp_name-{args.exp_tag}"
    if not os.path.exists(args.cache_root):
        os.makedirs(args.cache_root)
        with open(f"{args.cache_root}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    else:
        old_args = json.load(open(f"{args.cache_root}/args.json", "r"))
        # 判断args是否有变化
        if old_args != vars(args):
            print("Warning: args.json文件已存在，但是args参数发生了变化，继续则将抹去原有的cache文件，是否继续？(y/n)", end="")
            if input() == "y":
                os.system(f"rm -rf {args.cache_root}")
                os.makedirs(args.cache_root)
                with open(f"{args.cache_root}/args.json", "w") as f:
                    json.dump(vars(args), f, indent=4)
            else:
                exit()
    print(f"本次实验的cache路径为：{args.cache_root}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Context: Parser docs, build retrieval engine, and retrieve and rerank the context.')
    parser.add_argument('--data_root', type=list, default=["./data/0715"],
                        help="The root path of the docs. 可以是一个目录，也可以是多个目录")
    parser.add_argument('--node_parser', type=str, default="sentenswindows",
                        help="The parser of the nodes, can be simple or sentenswindows")
    parser.add_argument('--chunk_size', type=int, default=300)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--overlap', type=int, default=150)
    parser.add_argument('--emb_model_path', type=str, default='./maidalun/bce-embedding-base_v1')
    parser.add_argument('--emb_batch_size', type=int, default=256)
    parser.add_argument('--rr_model_path', type=str, default='./maidalun/bce-reranker-base_v1')
    parser.add_argument('--bm25_topk', type=int, default=50)
    parser.add_argument('--emb_topk', type=int, default=50)
    parser.add_argument('--cache_root', type=str, default='./cache')
    parser.add_argument('--debug_query_path', type=str, default='./questions/debug.jsonl',
                        help="debug 问题文件路径")
    parser.add_argument('--submit_query_path', type=str, default='./questions/question.jsonl',
                        help="submit 问题文件路径")
    parser.add_argument('--distance', type=str, default='COSINE',
                        help="The distance metric for vector similarity, can be COSINE or DOT")
    parser.add_argument('--keywords_file_path', type=str, default='./src/keywords_from_docs.txt',
                        help="关键词文件路径")
    parser.add_argument('--cut_for_search', type=bool, default=False,
                        help="是否用jieba的cut for search分词")
    parser.add_argument('--exp_tag', type=str, default=None,
                        help="The tag of the experiment")
    args = parser.parse_args()
    
    build_cache(args)
    
    device = infer_dvice()
    
    docs_nodes = Docs_Nodes(root_path = args.data_root,
                             docs_names = ["director", "emsplus", "rcp", "umac"],
                             node_parser = args.node_parser,
                             chunk_size = args.chunk_size,
                             window_size = args.window_size,
                             overlap = args.overlap,
                             cache_root = args.cache_root)
    
    emb_model = EmbModel(model_name = args.emb_model_path,
                         embed_batch_size = args.emb_batch_size,
                         node_space = docs_nodes.nodes_space,
                         device = device,
                         cache_root = args.cache_root)
    
    vectors_store = VectorStore(ids_dict = emb_model.ids_dict,
                                node_embs = emb_model.nodes_embeddings_dict,
                                emb_dim = emb_model.emb_dim,
                                distance = args.distance)
    
    bm25 = BM25(nodes_space = docs_nodes.nodes_space,
                keywords_file_path = args.keywords_file_path,
                cut_for_search = args.cut_for_search,
                cache_root = args.cache_root)
    
    rr_model = ReRank_Model(model_name = args.rr_model_path,
                            device = device)
    
    retriever = Retriever(docs_nodes = docs_nodes,
                          emb_model = emb_model,
                          rr_model = rr_model,
                          bm25 = bm25,
                          vector_store = vectors_store,
                          device = device,
                          cache_root = args.cache_root)
    
    # 加载问题
    debug_queries = load_query(args.debug_query_path)
    submit_queries = load_query(args.submit_query_path)
    
    retriever.batch_retrieval_and_rerank(debug_queries,
                                         submit_queries,
                                         bm25_topk=args.bm25_topk,
                                         emb_topk=args.emb_topk)