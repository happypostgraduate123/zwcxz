# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/07/18 09:10:50
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''

import torch
import hashlib

def infer_dvice()->str:
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    return device

def compute_hash(content: str) -> str:
    """计算文本内容的哈希值"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()