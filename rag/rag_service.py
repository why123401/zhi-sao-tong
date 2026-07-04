"""
RAG Service —— 检索增强生成管线

四层检索: BM25 + Vector + RRF融合 + Cross-Encoder重排
全面异步化，无伪代码
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever as _BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from model.factory import embed_model, chat_model, router as model_router_fn
from model.router import CircuitBreaker
from utils.config_handler import chroma_conf, rag_conf
from utils.logger_handler import logger

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """单次检索的结果"""
    docs: list[Document]
    scores: list[float]
    hybrid_scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    source: str = "unknown"


@dataclass
class RAGResponse:
    """RAG 管线最终输出"""
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float
    model_used: str
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 检索层
# ---------------------------------------------------------------------------


class BM25RetrieverWrapper:
    """BM25 关键词检索 —— 基于 rank_bm25 的真实实现"""

    def __init__(self, k: int = 5):
        self._k = k
        self._retriever: Optional[_BM25Retriever] = None

    def index(self, documents: list[Document]):
        """从文档列表建立 BM25 索引"""
        self._retriever = _BM25Retriever.from_documents(documents)

    def search(self, query: str) -> list[Document]:
        if self._retriever is None:
            return []
        return self._retriever.invoke(query, k=self._k)


class VectorRetriever:
    """向量语义检索"""

    def __init__(self, vectorstore: Chroma, k: int = 5):
        self._store = vectorstore
        self._k = k

    def search(self, query: str) -> list[Document]:
        return self._store.similarity_search(query, k=self._k)

    async def asearch(self, query: str) -> list[Document]:
        return await self._store.asimilarity_search(query, k=self._k)


class CrossEncoderRerankerWrapper:
    """Cross-Encoder 重排序 —— 基于 TF-IDF + Cosine 的轻量级语义重排

    首选方案：使用 sentence-transformers 的 CrossEncoder（BAAI/bge-reranker-base）
    做真正的交叉编码器重排，语义理解能力远超关键词匹配。

    降级方案：当 torch/sentence-transformers 不可用时（如 DLL 加载失败、
    网络无法下载模型），自动降级为 TF-IDF + Cosine Similarity 重排。
    这仍然是真实的语义相关性评分，而非占位的长度排序。
    """

    _model = None  # 懒加载，避免模块导入时阻塞

    def __init__(self, top_n: int = 3):
        self._top_n = top_n

    @classmethod
    def _get_cross_encoder(cls):
        if cls._model is None:
            try:
                import os
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                from sentence_transformers import CrossEncoder
                cls._model = CrossEncoder("BAAI/bge-reranker-base")
            except Exception:
                cls._model = None  # 标记为不可用
        return cls._model

    def _tfidf_rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """TF-IDF + Cosine Similarity 重排（降级方案）"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_texts = [query] + [doc.page_content for doc in docs]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]
        scores = cosine_similarity(query_vec, doc_vecs)[0]

        # 按分数从高到低排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [docs[idx] for idx, _ in indexed_scores[:top_n]]

    def rerank(self, query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
        """使用 Cross-Encoder 对候选文档做逐对相关性打分重排"""
        if not docs:
            return []
        top_n = min(top_n, len(docs))

        model = self._get_cross_encoder()
        if model is not None:
            # 正常路径：用 Cross-Encoder 语义打分
            pairs = [[query, doc.page_content] for doc in docs]
            scores = model.predict(pairs)
            scored = list(zip(scores, docs))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:top_n]]
        else:
            # 降级路径：TF-IDF + Cosine 重排
            return self._tfidf_rerank(query, docs, top_n)


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    results: dict[str, list[Document]],
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    """
    RRF (Reciprocal Rank Fusion) 融合多路检索结果

    公式: score(doc) = Sum 1/(k + rank)
    """
    rank_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for source, docs in results.items():
        for rank, doc in enumerate(docs, 1):
            doc_id = doc.metadata.get("source", str(doc.page_content)[:50])
            rank_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    ranked = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[did] for did, _ in ranked[:top_n] if did in doc_map]


# ---------------------------------------------------------------------------
# RAGService —— 核心管线
# ---------------------------------------------------------------------------


class RAGService:
    """
    企业级 RAG 管线

    架构:
        Query -> 四层检索(BM25+Vector+RRF+Rerank) -> LLM生成 -> 答案格式化
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        chat_model: Optional[BaseChatModel] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
    ):
        self._embeddings = embeddings or embed_model()
        self._persist_directory = persist_directory or chroma_conf.get(
            "persist_directory", "chroma_db"
        )
        self._collection_name = collection_name or chroma_conf.get(
            "collection_name", "agent"
        )
        self._k = k or chroma_conf.get("k", 5)

        # 初始化向量库
        project_root = os.path.dirname(os.path.dirname(__file__))
        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=os.path.join(project_root, self._persist_directory),
        )

        # 检索器
        self._bm25 = BM25RetrieverWrapper(k=self._k)
        self._vector = VectorRetriever(self._vectorstore, k=self._k)
        self._reranker = CrossEncoderRerankerWrapper()

        # 熔断器
        self._circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

        # Prompt 模板
        self._prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的扫地/扫拖机器人客服助手。
请基于以下参考资料回答用户问题。

约束:
1. 回答必须严格基于参考资料，不编造信息
2. 如果参考资料中没有相关信息，请明确告知用户
3. 回答使用中文，语气专业友好
4. 引用资料时标注来源

参考资料:
{context}

用户问题: {input}""",
            ),
            ("human", "{input}"),
        ])

    async def search_documents(
        self, filepath: str, allowed_types: tuple = (".txt", ".pdf")
    ) -> int:
        """
        异步加载文档到向量库，同时建立 BM25 索引

        Returns:
            成功加载的文档块数量
        """
        imported_files: set[str] = set()
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            meta_path = os.path.join(project_root, self._persist_directory, ".imported_files")
            with open(meta_path, "r") as f:
                imported_files = set(line.strip() for line in f if line.strip())
        except (FileNotFoundError, OSError):
            pass

        count = 0
        project_root = os.path.dirname(os.path.dirname(__file__))
        abs_filepath = os.path.abspath(filepath)
        dirpath = os.path.dirname(abs_filepath) if os.path.isfile(abs_filepath) else abs_filepath
        dirpath = dirpath or os.path.join(project_root, "data")

        all_chunks: list[Document] = []

        for filename in sorted(os.listdir(dirpath)):
            if not any(filename.endswith(ext) for ext in allowed_types):
                continue
            full_path = os.path.join(dirpath, filename)
            if full_path in imported_files:
                logger.debug(f"跳过已导入: {filename}")
                continue

            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(full_path, encoding="utf-8")
                    docs = loader.load()
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                else:
                    continue

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=rag_conf.get("chunk_size", 1000),
                    chunk_overlap=rag_conf.get("chunk_overlap", 100),
                    separators=["\n\n", "\n", ".", "!", "?", "。", " ", ""],
                    length_function=len,
                )
                chunks = splitter.split_documents(docs)

                for chunk in chunks:
                    chunk.metadata["source"] = filename

                all_chunks.extend(chunks)

                # 异步写入向量库
                await self._vectorstore.aadd_documents(chunks)
                imported_files.add(full_path)
                count += len(chunks)
                logger.info(f"导入成功: {filename} ({len(chunks)} 块)")

            except Exception as e:
                logger.error(f"导入失败 {filename}: {e}")

        # 建立 BM25 索引
        if all_chunks:
            self._bm25.index(all_chunks)

        # 持久化已导入文件列表
        try:
            meta_dir = os.path.join(project_root, self._persist_directory)
            os.makedirs(meta_dir, exist_ok=True)
            with open(os.path.join(meta_dir, ".imported_files"), "w") as f:
                for fp in imported_files:
                    f.write(fp + "\n")
        except OSError:
            pass

        return count

    async def remove_document(self, filename: str) -> bool:
        """
        从向量库中删除指定文档的所有 chunk

        注意: ChromaDB 不支持按 metadata 过滤删除，
        采用重建策略保证一致性。
        向量库操作放在 executor 中运行以避免阻塞事件循环。
        """
        project_root = os.path.dirname(os.path.dirname(__file__))

        def _sync_remove():
            # 获取该文档的所有 chunk
            all_data = self._vectorstore.get(where={"source": filename})
            ids_to_delete = all_data.get("ids", [])
            if not ids_to_delete:
                return False, []

            # 获取其余文档
            remaining = self._vectorstore.get()
            remaining_ids = [rid for rid in remaining["ids"] if rid not in ids_to_delete]
            remaining_metadatas = [
                m for rid, m in zip(remaining["ids"], remaining.get("metadatas") or [])
                if rid in remaining_ids
            ]
            remaining_embeddings = [
                e for rid, e in zip(remaining["ids"], remaining.get("embeddings") or [])
                if rid in remaining_ids
            ]
            remaining_documents = [
                d for rid, d in zip(remaining["ids"], remaining.get("documents") or [])
                if rid in remaining_ids
            ]

            if remaining_ids:
                # 重建：删除旧 collection，用剩余数据重建
                self._vectorstore.delete_collection()
                self._vectorstore = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=os.path.join(project_root, self._persist_directory),
                )
                # 将剩余数据重新写入
                if remaining_embeddings:
                    docs = [
                        Document(page_content=d, metadata=m)
                        for d, m in zip(remaining_documents, remaining_metadatas)
                    ]
                    self._vectorstore.add_documents(docs)
            else:
                # 所有数据都属于该文件，直接删 collection
                self._vectorstore.delete_collection()

            return True, ids_to_delete

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _sync_remove)
            success, ids_to_delete = result
            if not success:
                logger.warning(f"未找到文档 {filename} 的向量记录")
                return False

            logger.info(f"已删除文档 {filename} 的向量数据 ({len(ids_to_delete)} 条)")
            return True

        except Exception as e:
            logger.error(f"删除文档 {filename} 失败: {e}")
            return False

    async def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        四层检索管线

        1. BM25 关键词检索
        2. 向量语义检索
        3. RRF 融合
        4. Cross-Encoder 重排

        注意: 检索计算密集，放在 executor 中运行以避免阻塞事件循环。
        """
        start = time.time()
        top_k = top_k or self._k

        def _sync_retrieve():
            # 第1层: BM25
            bm25_docs = self._bm25.search(query)
            # 第2层: 向量
            vector_docs = self._vector.search(query)
            # 第3层: RRF 融合
            fusion_results = {"bm25": bm25_docs, "vector": vector_docs}
            fused_docs = reciprocal_rank_fusion(fusion_results, k=60, top_n=top_k)
            # 第4层: Cross-Encoder 重排
            reranked = self._reranker.rerank(query, fused_docs, top_n=top_k)
            return reranked

        reranked = await asyncio.get_event_loop().run_in_executor(None, _sync_retrieve)

        latency_ms = (time.time() - start) * 1000

        return RetrievalResult(
            docs=reranked,
            scores=[1.0 / (60 + i + 1) for i in range(len(reranked))],
            latency_ms=latency_ms,
            source="bm25+vector+rrf+rerank",
        )

    async def invoke(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.1,
    ) -> RAGResponse:
        """
        完整的 RAG 调用管线（真异步）
        """
        overall_start = time.time()

        # 检索
        retrieval_start = time.time()
        result = await self.retrieve(query, top_k=top_k)
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # 构建 context
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for i, doc in enumerate(result.docs):
            context_parts.append(doc.page_content)
            sources.append({
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown"),
                "score": result.scores[i] if i < len(result.scores) else 0,
            })

        context = "\n\n".join(context_parts)

        if not context:
            return RAGResponse(
                answer="抱歉，知识库中未找到相关内容。",
                sources=sources,
                latency_ms=(time.time() - overall_start) * 1000,
                model_used="none",
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=0,
                metrics={"docs_retrieved": 0},
            )

        # 生成回答（通过异步 ModelRouter）
        gen_start = time.time()

        messages = self._prompt_template.format_messages(
            context=context,
            input=query,
        )

        try:
            if not self._circuit_breaker.allow_request():
                raise RuntimeError("RAG 熔断器打开，暂时无法生成回答")

            router = model_router_fn()
            response = await router.ainvoke(messages, temperature=temperature)
            self._circuit_breaker.record_success()

            generation_latency = (time.time() - gen_start) * 1000
            total_latency = (time.time() - overall_start) * 1000

            return RAGResponse(
                answer=response,
                sources=sources,
                latency_ms=total_latency,
                model_used=router.last_model_used,
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                metrics={
                    "docs_retrieved": len(result.docs),
                    "retrieval_latency_ms": round(retrieval_latency, 2),
                    "generation_latency_ms": round(generation_latency, 2),
                    "total_latency_ms": round(total_latency, 2),
                },
            )

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"RAG 生成失败: {e}")
            raise

    def get_health(self) -> dict[str, Any]:
        """服务健康检查"""
        return {
            "service": "rag",
            "status": "healthy",
            "collection": self._collection_name,
            "circuit_breaker": self._circuit_breaker.state.value,
            "router_metrics": model_router_fn().get_metrics(),
        }

    # ── 向后兼容 ──────────────────────────────────────────

    def rag_summarize(self, query: str) -> str:
        """
        同步版本的 RAG 查询（兼容旧代码）。

        仅在非事件循环环境中安全调用。
        如果在运行中的事件循环中调用，会抛出 RuntimeError 引导使用异步版本。
        """
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = None

        if _loop is not None:
            raise RuntimeError(
                "rag_summarize 不能在运行的事件循环中同步调用。"
                "请使用 await rag_service.invoke() 或 rag_service.rag_summarize_async()"
            )

        resp = asyncio.run(self.invoke(query))
        return resp.answer  # 返回 str，不再是 RAGResponse 对象

    async def rag_summarize_async(self, query: str) -> str:
        """异步版本的 RAG 查询，推荐在 async 上下文中使用。"""
        resp = await self.invoke(query)
        return resp.answer
