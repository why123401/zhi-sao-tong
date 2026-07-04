"""向量库管理服务（已迁移至 rag/rag_service.py）

此文件保留向后兼容。新的 RAGService 在 rag/rag_service.py 中实现，
包含四层检索、异步化、RAG 生成管线等完整功能。

如需直接使用向量库操作，可通过 RAGService._vectorstore 访问。
"""

from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreManager:
    """
    向量库基础管理（薄封装，实际功能在 RAGService 中）。

    保留此类以兼容旧代码中直接操作向量库的场景。
    """

    def __init__(self) -> None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        persist_dir = os.path.join(
            project_root, chroma_conf.get("persist_directory", "chroma_db")
        )
        self._store = Chroma(
            collection_name=chroma_conf.get("collection_name", "agent"),
            embedding_function=embed_model(),
            persist_directory=persist_dir,
        )
        self._spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf.get("chunk_size", 1000),
            chunk_overlap=chroma_conf.get("chunk_overlap", 100),
            separators=chroma_conf.get(
                "separators", ["\n\n", "\n", ".", "!", "?", "。", " ", ""]
            ),
            length_function=len,
        )

    def get_retriever(self, k: int | None = None):
        """返回 LangChain retriever"""
        k = k or chroma_conf.get("k", 3)
        return self._store.as_retriever(search_kwargs={"k": k})

    @property
    def store(self) -> Chroma:
        """直接访问底层 Chroma 实例"""
        return self._store

    def split_text(self, text: str) -> list[Document]:
        """将文本分割为文档块"""
        doc = Document(page_content=text)
        return self._spliter.split_documents([doc])

    def add_texts(self, texts: list[str], metadata: dict | None = None) -> None:
        """批量添加文本到向量库"""
        docs = [Document(page_content=t, metadata=metadata or {}) for t in texts]
        self._store.add_documents(docs)
        logger.info(f"[VectorStoreManager] 添加了 {len(texts)} 条文本")

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """相似度搜索"""
        return self._store.similarity_search(query, k=k)
