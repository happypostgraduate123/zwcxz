---
license: Apache License 2.0
---
数据集文件元信息以及数据文件，请浏览“数据集文件”页面获取。

#### 数据集说明

复赛所用到的文档与初赛一致，请从https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset  下载

#### 文件说明

`debug.jsonl`用于选手在复赛阶段对方案进行debug，共计50题，其中包含的字段有

```
id: 题目的id
query：题目
answer：由专家标注的答案
document：答案所在文档
```

`question.jsonl`用于复赛的评比，共计103题，其中包含的字段有

```
id: 题目的id
query：题目
document：答案所在文档
```

