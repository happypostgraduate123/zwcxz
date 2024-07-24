# -*- encoding: utf-8 -*-
'''
@File    :   vector_store.py
@Time    :   2024/07/09 22:55:11
@Author  :   Fei Gao
@Contact :   feigao.sc@gmail.com
Beijing, China
'''
from typing import Dict
from qdrant_client import models, QdrantClient



class VectorStore:
    def __init__(self,
                 ids_dict: Dict,
                 node_embs: Dict,
                 emb_dim: int,
                 distance: str = "COSINE"):
        self.ids_dict = ids_dict
        self.node_embs = node_embs
        self.emb_dim = emb_dim
        self.distance = self.infer_distance(distance)
        self.qdrant_client = self.build_vector_store()

    def infer_distance(self, distance: str) -> models.Distance:
        assert distance in ["COSINE", "DOT"]
        if distance == "COSINE":
            return models.Distance.COSINE
        elif distance == "DOT":
            return models.Distance.DOT
        else:
            raise ValueError(f"Unknown distance: {distance}")
    
    def build_vector_store(self) -> QdrantClient:
        print("Building vector store in Qdrant with distance: ", self.distance)
        qdrant_client = QdrantClient(location=":memory:")
        for name in self.ids_dict.keys():
            ids = self.ids_dict[name]
            embs = self.node_embs[name]
            collection_name = name + "_collection"
            qdrant_client.recreate_collection(collection_name=collection_name,
                                              vectors_config=models.VectorParams(size = self.emb_dim,
                                                                                 distance = self.distance))
            qdrant_client.upload_collection(collection_name=collection_name,
                                            vectors=embs,
                                            ids=ids,
                                            batch_size=600)
        return qdrant_client