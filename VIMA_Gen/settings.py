"""
VIMA_Gen 集中式配置加载。

把「密钥 / 接口地址」与「运行参数」从代码里抽出来：

* **敏感项与接口配置** → 项目根目录 ``.env``（模板见 ``.env_example``，已被 .gitignore 忽略）
* **非敏感运行参数** → ``VIMA_Gen/config.yaml``

优先级（高 → 低）::

    CLI 参数  >  .env / 环境变量  >  config.yaml  >  本文件内置默认值

``.env`` 查找顺序为 ``<项目根>/.env`` → ``<项目根>/VIMA_Gen/.env``；
已存在的 shell 环境变量不会被覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from omegaconf import OmegaConf

__all__ = [
    "LLMConfig",
    "EmbeddingConfig",
    "RunConfig",
    "load_config",
    "load_dotenv_files",
    "resolve_path",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LOCAL_EMBEDDING_MODEL",
]

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent

DEFAULT_CONFIG_PATH = _THIS_DIR / "config.yaml"
DOTENV_CANDIDATES = (_ROOT_DIR / ".env", _THIS_DIR / ".env")

# 环境变量名
ENV_API_KEY = "OPENAI_API_KEY"
ENV_BASE_URL = "OPENAI_BASE_URL"
ENV_MODEL = "VIMA_LLM_MODEL"
ENV_TEMPERATURE = "VIMA_TEMPERATURE"
ENV_DEVICE = "VIMA_DEVICE"
# 嵌入后端与 chat 分离：DeepSeek 等网关只有 chat，没有 /embeddings
ENV_EMBEDDING_PROVIDER = "VIMA_EMBEDDING_PROVIDER"
ENV_EMBEDDING_MODEL = "VIMA_EMBEDDING_MODEL"
ENV_EMBEDDING_BASE_URL = "VIMA_EMBEDDING_BASE_URL"
ENV_EMBEDDING_API_KEY = "VIMA_EMBEDDING_API_KEY"
ENV_LOCAL_EMBEDDING_MODEL = "VIMA_LOCAL_EMBEDDING_MODEL"

# 内置默认值（config.yaml 缺失或被删字段时兜底）
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """OpenAI 兼容接口配置。

    字段名与 langchain-openai 的 pydantic 字段保持一致，避免别名歧义：
    ``ChatOpenAI(model_name=..., temperature=..., openai_api_base=..., openai_api_key=...)``。
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE

    # -- 构造 langchain 客户端参数 -------------------------------------------

    def chat_kwargs(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """``ChatOpenAI`` 的构造参数。仅传入已配置的可选项，避免覆盖 SDK 默认行为。"""
        kwargs: dict[str, Any] = {
            "model_name": model or self.model,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.api_key:
            kwargs["openai_api_key"] = self.api_key
        if self.base_url:
            kwargs["openai_api_base"] = self.base_url
        return kwargs

    def require_api_key(self) -> str:
        """返回 API key；未配置时给出可操作的报错。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 API key。请复制模板并填入 OPENAI_API_KEY：\n"
                f"    cp {_ROOT_DIR / '.env_example'} {_ROOT_DIR / '.env'}\n"
                "然后编辑 .env（也可直接用环境变量 OPENAI_API_KEY）。"
            )
        return self.api_key


@dataclass
class EmbeddingConfig:
    """RAG 嵌入后端配置。

    ``provider="local"``（默认）用 sentence-transformers 本地模型，离线且无需第二个 key；
    ``provider="openai"`` 用独立的 OpenAI 兼容 ``/embeddings`` 接口。
    """

    provider: str = DEFAULT_EMBEDDING_PROVIDER
    local_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    model: str = DEFAULT_EMBEDDING_MODEL
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def is_local(self) -> bool:
        return (self.provider or "local").strip().lower() in (
            "local",
            "sentence-transformers",
            "st",
        )

    def openai_kwargs(self) -> dict[str, Any]:
        """``OpenAIEmbeddings`` 的构造参数（provider=openai 时使用）。"""
        if not self.base_url:
            raise RuntimeError(
                "provider=openai 需要显式配置嵌入服务地址：请在 .env 设置"
                " VIMA_EMBEDDING_BASE_URL 与 VIMA_EMBEDDING_API_KEY，"
                "或把 VIMA_EMBEDDING_PROVIDER 改回 local（离线，无需额外 key）。"
            )
        if not self.api_key:
            raise RuntimeError(
                "provider=openai 需要 VIMA_EMBEDDING_API_KEY"
                "（注意：与 chat 的 OPENAI_API_KEY 是分开配置的）。"
            )
        return {
            "model": self.model,
            "openai_api_base": self.base_url,
            "openai_api_key": self.api_key,
        }

    def describe(self) -> str:
        if self.is_local:
            return f"local:{self.local_model}"
        return f"openai:{self.model} @ {self.base_url}"


@dataclass
class RunConfig:
    """一次生成运行的全部参数。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # generation
    n: int = 1
    k: int = 5
    brief: Optional[str] = None
    save: bool = False

    # paths（相对路径一律相对 VIMA_Gen/ 解析，见 resolve_path）
    save_dir: str = "generated_tasks"
    run_results_dir: str = "run_results"
    failed_store_path: str = "failed_generations.json"

    # runtime
    device: Optional[str] = None

    def summary(self) -> str:
        """一行式配置摘要（不输出密钥内容）。"""
        return " | ".join(
            [
                f"model={self.llm.model}",
                f"embedding={self.embedding.describe()}",
                f"temperature={self.llm.temperature}",
                f"base_url={self.llm.base_url or '(default)'}",
                f"api_key={'set' if self.llm.api_key else 'MISSING'}",
                f"n={self.n}",
                f"k={self.k}",
                f"save={self.save}",
                f"device={self.device or 'auto'}",
            ]
        )


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def load_dotenv_files() -> list[Path]:
    """加载所有存在的 .env 文件（不覆盖已有环境变量）。返回实际加载的路径。"""
    loaded: list[Path] = []
    for path in DOTENV_CANDIDATES:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


def resolve_path(value: str | Path) -> Path:
    """把配置里的路径解析为绝对路径：相对路径基于 VIMA_Gen/ 目录。"""
    path = Path(value)
    return path if path.is_absolute() else (_THIS_DIR / path)


def _select(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """从 OmegaConf 节点安全取值（路径不存在时返回 default）。"""
    if cfg is None:
        return default
    try:
        value = OmegaConf.select(cfg, dotted_key)
    except Exception:  # pragma: no cover - 防御性
        return default
    return default if value is None else value


def load_config(config_path: Optional[str | Path] = None) -> RunConfig:
    """加载 .env + config.yaml，合并为一个 :class:`RunConfig`。

    Args:
        config_path: 配置文件路径（``--config``）。相对路径基于项目根目录解析。
    """
    load_dotenv_files()

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _ROOT_DIR / path
    cfg = OmegaConf.load(path) if path.is_file() else None

    # 环境变量优先于 config.yaml
    llm = LLMConfig(
        api_key=os.environ.get(ENV_API_KEY) or _select(cfg, "llm.api_key"),
        base_url=os.environ.get(ENV_BASE_URL) or _select(cfg, "llm.base_url"),
        model=os.environ.get(ENV_MODEL) or _select(cfg, "llm.model", DEFAULT_MODEL),
        temperature=float(
            os.environ.get(ENV_TEMPERATURE)
            or _select(cfg, "llm.temperature", DEFAULT_TEMPERATURE)
        ),
    )

    embedding = EmbeddingConfig(
        provider=os.environ.get(ENV_EMBEDDING_PROVIDER)
        or _select(cfg, "embedding.provider", DEFAULT_EMBEDDING_PROVIDER),
        local_model=os.environ.get(ENV_LOCAL_EMBEDDING_MODEL)
        or _select(cfg, "embedding.local_model", DEFAULT_LOCAL_EMBEDDING_MODEL),
        model=os.environ.get(ENV_EMBEDDING_MODEL)
        or _select(cfg, "embedding.model", DEFAULT_EMBEDDING_MODEL),
        base_url=os.environ.get(ENV_EMBEDDING_BASE_URL)
        or _select(cfg, "embedding.base_url"),
        api_key=os.environ.get(ENV_EMBEDDING_API_KEY)
        or _select(cfg, "embedding.api_key"),
    )

    return RunConfig(
        llm=llm,
        embedding=embedding,
        n=int(_select(cfg, "generation.n", 1)),
        k=int(_select(cfg, "retrieval.k", 5)),
        brief=_select(cfg, "generation.brief"),
        save=bool(_select(cfg, "generation.save", False)),
        save_dir=str(_select(cfg, "paths.save_dir", "generated_tasks")),
        run_results_dir=str(_select(cfg, "paths.run_results_dir", "run_results")),
        failed_store_path=str(
            _select(cfg, "paths.failed_store", "failed_generations.json")
        ),
        device=os.environ.get(ENV_DEVICE) or _select(cfg, "device"),
    )
