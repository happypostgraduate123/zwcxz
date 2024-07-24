# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/09 21:54:11
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''

# 导入基础库
import os
import re
import hashlib
import pickle as pkl
from typing import List, Callable
from types import SimpleNamespace

# 导入高级库
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser, SimpleNodeParser

def info_len(s: str) -> int:
    """自定义的字符串长度计算函数，移除空格等特殊字符。
    
    Args:
        s (str): 输入字符串

    Returns:
        int: 字符串长度
    """
    return len(re.findall(r'\w', s))

def split_by_line(CHUNK_SIZE: int) -> Callable[[str], List[str]]:
    def split(text: str, CHUNK_SIZE: int = CHUNK_SIZE) -> List[str]:
        """自定义的文本分割函数，按行分割并拼接文本块。
        
        Args:
            text (str): 输入文本        

        Returns:
            List[str]: 分割的块， 每个块的长度不超过 CHUNK_SIZE
        """
        blocks = re.split(r'\n', text)
        blocks = [block.strip() for block in blocks if block.strip()]
        
        result = []
        current_block = ""
        for block in blocks:
            if info_len(current_block + block + '\n') <= CHUNK_SIZE:
                current_block += block + '\n'
            else:
                result.append(current_block)
                current_block = block + '\n'
        if current_block:
            result.append(current_block)
        return result
    return split

def build_SentenceWindowNodeParser(chunk_size: int = 500, window_size: int = 3) -> SentenceWindowNodeParser:
    """构建一个基于行分割的 SentenceWindowNodeParser。

    Args:
        chunk_size (int, optional): 最大的分块长度. Defaults to 500.
        window_size (int, optional): 每个块前后窗口的大小. Defaults to 3.

    Returns:
        SentenceWindowNodeParser: 返回的 node_parser。
    """
    return SentenceWindowNodeParser.from_defaults(
        sentence_splitter=split_by_line(chunk_size),
        window_size=window_size,
        window_metadata_key='window',
        original_text_metadata_key='original_sentence')

def build_SimpleNodeParser(chunk_size: int = 1024,
                           overlap: int = 20) -> SimpleNodeParser:
    """构建一个简单的 SentenceSplitter。

    Returns:
        SimpleNodeParser: 返回的 node_parser。
    """
    return SimpleNodeParser(separator="\n",
                            chunk_size=chunk_size,
                            chunk_overlap=overlap,
                            paragraph_separator="\n\n",
                            secondary_chunking_regex="[^，。；！？]+[，。；！？]?")

def FilePathExtractor(doc, FILE_ROOT: List):
    try:
        path_parts = doc.metadata["file_path"].split("/")
        for root in FILE_ROOT:
            root_name = root.split('/')[-1]
            if root_name in path_parts:
                data_index = path_parts.index(root_name)
                doc.metadata["file_path"] = "/".join(path_parts[data_index+1:])
                break
    except ValueError as e:
        print(f"Error extracting file path: {e}")
    return doc


class Docs_Nodes:
    def __init__(self,
                 root_path: str,
                 docs_names: List[str],
                 node_parser: str = "sentenswindows",
                 chunk_size: int = 500,
                 window_size: int = 3,
                 overlap: int = 100,
                 cache_root: str = "./cache"):
        self.root_path = root_path
        self.docs_names = docs_names
        self.node_parser_name = node_parser
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.overlap = overlap
        self.cache_root = cache_root
        
        self.docs_space = self.load_docs()    
        self.node_parser = self.build_node_parser()
        self.nodes_space = self.parser_docs()
        
    def build_node_parser(self):
        if self.node_parser_name == "sentenswindows":
            node_parser = build_SentenceWindowNodeParser(chunk_size=self.chunk_size,
                                                         window_size=self.window_size)
            print("Building SentenceWindowNodeParser with chunk_size: {} and window_size: {}".format(self.chunk_size,
                                                                                                     self.window_size))
        elif self.node_parser_name == "simple":
            node_parser = build_SimpleNodeParser(chunk_size=self.chunk_size,
                                                 overlap=self.overlap)
            print("Building SimpleNodeParser with chunk_size: {} and overlap: {}".format(self.chunk_size,
                                                                                         self.overlap))
            
        else:
            raise ValueError("Node parser name {} not supported.".format(self.node_parser_name))
        return node_parser
        
    def load_docs(self) -> SimpleNamespace:
        """加载原始txt文档。

        Returns:
            SimpleNamespace: 文档空间
        """
        docs_space = SimpleNamespace()
        for name in self.docs_names:
            docs = []
            for root_path in self.root_path:
                reader = SimpleDirectoryReader(input_dir=os.path.join(root_path, name),
                                               required_exts=[".txt"],
                                               recursive=True)
                docs += reader.load_data()
            setattr(docs_space, name, docs)
        return docs_space

    def parser_docs(self) -> SimpleNamespace:
        if self.__cache_exist:
            print("文档解析Cache从{}加载。".format(self.__cache_save_path))
            nodes_space = self.__load_nodes_cache()
        else:
            print("Cache没有找到，开始解析文档。")
            nodes_space = SimpleNamespace()
            for name in self.docs_names:
                docs = getattr(self.docs_space, name)
                docs = [FilePathExtractor(doc, self.root_path) for doc in docs]
                nodes = self.node_parser.get_nodes_from_documents(docs)
                self.set_nodes_attributes(nodes)
                setattr(nodes_space, name, nodes)
                print("Document: {:^8} Number of documents: {:<5} Total nodes: {:<5}".format(name, len(docs), len(nodes)))
            self.__save_nodes_cache(nodes_space)
        return nodes_space
    
    def set_nodes_attributes(self, nodes: List[SimpleNamespace]):
        """设置节点的属性。用来区分检索和重排用到的文本。
        """
        for i, node in enumerate(nodes):
            if self.node_parser_name == "sentenswindows":
                node.metadata["text_for_retrieval"] = node.text
                node.metadata["text_for_rerank"]    = node.text
                node.metadata["text_for_answer"]    = node.metadata["window"]
            elif self.node_parser_name == "simple":
                node.metadata["text_for_retrieval"] = node.text
                node.metadata["text_for_rerank"]    = node.text
                node.metadata["text_for_answer"]    = node.text
            else:
                raise ValueError("Node parser name {} not supported.".format(self.node_parser_name))
    
    def __save_nodes_cache(self, nodes_space: SimpleNamespace):
        with open(self.__cache_save_path, "wb") as f:
            pkl.dump(nodes_space, f)

    def __load_nodes_cache(self):
        with open(self.__cache_save_path, "rb") as f:
            return pkl.load(f)
            
    @property
    def __cache_save_path(self):
        return os.path.join(self.cache_root, "nodes_space.pkl")
    
    @property
    def __cache_exist(self):
        return os.path.exists(self.__cache_save_path)