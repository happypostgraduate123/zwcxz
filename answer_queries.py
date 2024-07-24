# -*- encoding: utf-8 -*-
'''
@File    :   answer_queries.py
@Time    :   2024/07/20 00:32:59
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''
import warnings
warnings.filterwarnings('ignore')
import argparse

import os
import pickle as pkl
import jsonlines
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.QA import load_query, QA, load_query_source

def load_retrieval_rerank(cache_path:str):
    debug_path = os.path.join(cache_path, "retrieval_rerank_results_debug.pkl")
    with open(debug_path, "rb") as f:
        debug_retreival_rerank = pkl.load(f)
    
    submit_path = os.path.join(cache_path, "retrieval_rerank_results_submit.pkl")
    with open(submit_path, "rb") as f:
        submit_retreival_rerank = pkl.load(f)
    
    return debug_retreival_rerank, submit_retreival_rerank


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Answer queries by saved retrievals and LLM.")
    parser.add_argument('--debug_query_path', type=str, default='./questions/debug.jsonl',
                        help="Debug 文件路径")
    parser.add_argument('--debug_query_source_path', type=str, default='./questions/id2source.json',
                        help="Debug query source 文件路径")
    parser.add_argument('--submit_query_path', type=str, default='./questions/question.jsonl',
                        help="需要submit的query文件路径")
    parser.add_argument('--cache_path', type=str, default=None,
                        help="所需的cache文件路径")
    parser.add_argument('--max_retrieval_num', type=int, default=15,
                        help="最多使用的检索结果数量")
    parser.add_argument('--llm_concurrent', type=int, default=5)
    parser.add_argument('--llm_name', type=str, default="glm",
                        help="使用的llm模型名称, glm 或者 deepseek")
    
    args = parser.parse_args()
    
    # 加载query
    debug_queries = load_query(args.debug_query_path)
    debug_query_source = load_query_source(args.debug_query_source_path)
    
    # 加载retrieval_rerank的结果
    debug_retreival_rerank, submit_retreival_rerank = load_retrieval_rerank(args.cache_path)
    
    # 计算 debug 数据集上的检索性能
    QA_debug = QA(queries=debug_queries,
                  retrievals=debug_retreival_rerank,
                  id2source=debug_query_source,
                  llm_name=args.llm_name)
    QA_debug.retrieval_performence(cache_path=args.cache_path)
    
    print("debug数据集上的检索性能计算完毕, 是否继续生成submit结果？(y/n)", end=" ")
    if input() != "y":
        exit()
    
    submit_queries = load_query(args.submit_query_path)
    # submit_queries = submit_queries[:5]
    
    QA_submit = QA(queries=submit_queries,
                   retrievals=submit_retreival_rerank,
                   llm_name=args.llm_name)
    
    
    # 生成llm回答
    llm_answers, used_context = {}, {}
    with ThreadPoolExecutor(max_workers=args.llm_concurrent) as executor:        
        futures = []
        for id in [q['id'] for q in submit_queries]:
            futures.append(executor.submit(QA_submit.answer, id, args.max_retrieval_num))

        for future in tqdm(as_completed(futures), total=len(futures), desc="生成回答中"):
            id, context, answer = future.result()
            id = int(id)
            llm_answers[id] = answer
            used_context[id] = context
            
    with open(os.path.join(args.cache_path, "llm_answers.pkl"), "wb") as f:
        pkl.dump(llm_answers, f)
    
    # 将used_context按照id排序
    used_context = {k: used_context[k] for k in sorted(used_context.keys())}
    with open(os.path.join(args.cache_path, "used_context.json"), "w", encoding="utf-8") as f:
        json.dump(used_context, f, ensure_ascii=False, indent=4)
    
    # 保存结果
    answers = []
    for id, ans in llm_answers.items():
        answer_text = ans.text if ans is not None else ""
        answers.append(
            {"id": id, "query": submit_queries[id-1]["query"], "answer": answer_text}
        )
    
    # 按照 id 排序
    answers = sorted(answers, key=lambda x: x['id'])
    
    # 保存到本地，未处理回答失败的情况
    with jsonlines.open(os.path.join(args.cache_path, "submit_result_raw.jsonl"), "w") as json_file:
        json_file.write_all(answers)
    
    # 判断答案是否合理，尝试重新回答
    for ans in answers:
        cnt = 0
        while ans['answer']  == "" or len(ans['answer']) < 3:
            print("重新回答：", ans['id'])
            _, context, answer = QA_submit.answer(ans['id'], args.max_retrieval_num)
            ans['answer'] = answer.text
            cnt += 1
            if cnt > 5:
                print("回答失败：", ans['id'])
        
     # 保存到本地
    with jsonlines.open(os.path.join(args.cache_path, "submit_result.jsonl"), "w") as json_file:
        json_file.write_all(answers)
    
   