## 本方案的目录结构
```
├── answer_queries.py                           # 召回分析、答案生成
├── build_context.py                            # 分解文档、嵌入文档、检索引擎、检索结果、保存所有的Cache
├── cache                                       # cache目录（自动生成
├── data                                        # 放数据的目录，可以放不同版本的数据
├── maidalun                                    # 模型目录
├── questions
│   ├── aiops24-finals.json
│   ├── debug.jsonl                             
│   ├── Debug答案溯源-工作表1.csv
│   ├── id2source.json
│   ├── id2source.jsonl
│   ├── question.jsonl
│   ├── README.md
│   └── 提取源文件列表.ipynb
├── README.md
├── src
│   ├── BM25.py             # 构建BM25关键词检索
│   ├── Config.py           # 存储一些Prompt和llm api key
│   ├── __init__.py
│   ├── Models.py           # 构建嵌入和rerank模型
│   ├── __pycache__
│   ├── QA.py               # 利用存储好的召回文本，分析召回精度、生成答案等
│   ├── Retriever.py        # 检索器
│   ├── TextPrcess.py       # 所有的文档处理的代码
│   ├── Utils.py            
│   └── VectorStore.py      # 构建向量索引
└── submit.py
```

## 1. 预训练模型下载
本方案使用网易的BCE嵌入和重排模型，通过如下方式下载：
```
git clone https://www.modelscope.cn/maidalun/bce-embedding-base_v1.git maidalun/bce-embedding-base_v1
git clone https://www.modelscope.cn/maidalun/bce-reranker-base_v1.git maidalun/bce-reranker-base_v1
```

## 2. 运行本项目
- 第一步，准备数据和预训练模型。
- 第二步，阅读 build_context.py 的参数说明，按需设置，然后运行下面，用来处理文档、嵌入、召回，rerank等，直到保存检索的结果。
```
$ python build_context.py --exp_tag 
```
其中 exp_tag 用来区分不同实验的cache存储路径。注意，如果修改了原始数据或者关键词列表等，请删除cache，或者设置不同的exp_tag，以免cache导致实验结果出错
- 第三步，利用检索的结果分析召回效果、生成答案。请先阅读answer_queries的参数说明，并做相应的设置。特别是需要将cache路径设置为你想要用的cache。
```
$ python answer_queries.py
```

***几乎所有的信息都会打印出来，可以仔细查看cache或者结果的存储位置***
