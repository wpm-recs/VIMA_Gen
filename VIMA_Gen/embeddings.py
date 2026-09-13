"""
嵌入后端（RAG 检索器用的向量化模型）。

支持两种 provider：

* ``local``（默认）—— ``sentence-transformers``，**完全离线**，无需第二个 API key，
  且会自动跑在 :func:`vima_bench.tasks.utils.device.get_device` 解析出的设备上（有 CUDA 用 GPU）。
* ``openai`` —— 任意 OpenAI 兼容的 ``/embeddings`` 接口。

为什么默认 ``local``：很多 OpenAI 兼容网关只提供 chat 而**不提供 embeddings**
（例如 DeepSeek 的 ``/embeddings`` 返回 404）。默认本地化可以让整个流水线只依赖一个 chat key。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.embeddings import Embeddings

__all__ = ["LocalSentenceTransformerEmbeddings", "build_local_embeddings"]


class LocalSentenceTransformerEmbeddings(Embeddings):
    """把 ``sentence-transformers`` 包装成 LangChain 的 ``Embeddings`` 接口。

    Args:
        model_name: HuggingFace / sentence-transformers 的模型 id，
            例如 ``sentence-transformers/all-MiniLM-L6-v2``。
        device: ``"cuda"`` / ``"cuda:0"`` / ``"cpu"``；``None`` 表示用当前进程配置的设备。
        batch_size: 编码批大小。
        normalize_embeddings: 是否做 L2 归一化（配合 FAISS 内积/L2 距离使用，建议开启）。
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        # 延迟导入：避免仅 import 本模块就把 transformers 拉起来
        from sentence_transformers import SentenceTransformer

        from vima_bench.tasks.utils.device import get_device

        self.model_name = model_name
        self.device = str(device) if device is not None else str(get_device())
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name, device=self.device)

    # -- LangChain Embeddings 接口 -------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __repr__(self) -> str:  # pragma: no cover - 仅用于日志
        return (
            f"LocalSentenceTransformerEmbeddings(model={self.model_name!r}, "
            f"device={self.device!r})"
        )


def build_local_embeddings(
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> LocalSentenceTransformerEmbeddings:
    """便捷工厂。"""
    return LocalSentenceTransformerEmbeddings(
        model_name=model_name, device=device, batch_size=batch_size
    )
