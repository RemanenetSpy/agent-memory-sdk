"""Persistent agentic memory with semantic retrieval and restore/replay/verify decisions."""

from agent_memory.backends import (
    BackendContext,
    available_backends,
    register_backend,
)
from agent_memory.benchmark import BenchmarkResult, format_benchmark_report, run_benchmark
from agent_memory.benchmarks.harness import (
    BenchmarkDataset,
    BenchmarkHarness,
    HarnessConfig,
    HarnessResult,
)
from agent_memory.confidence import ConfidenceEvent, ConfidenceLearner, ConfidenceUpdate
from agent_memory.decision import DecisionEngine
from agent_memory.entity_extractor import EntityExtractor, ExtractedMemory
from agent_memory.eval import EvalDataset, EvalResult, format_eval_report, run_eval, run_eval_suite
from agent_memory.exceptions import (
    AgentMemoryError,
    BackendConnectionError,
    ConfigurationError,
    DecisionError,
    ExtractionError,
    MemoryNotFoundError,
    MemoryStoreError,
    MemoryWriteError,
    RetrievalError,
)
from agent_memory.graph import MemoryEdge, MemoryGraph
from agent_memory.knowledge_graph import (
    CompositeExtractionStrategy,
    EntityExtractionStrategy,
    EntityType,
    ExtractionResult,
    GraphBuilder,
    KnowledgeEntity,
    KnowledgeGraph,
    KnowledgeRelation,
    PatternExtractionStrategy,
    RelationType,
    SpacyExtractionStrategy,
)
from agent_memory.logging_config import configure_debug_logging, get_logger
from agent_memory.manager import Memory, MemoryManager
from agent_memory.models import (
    MemoryAction,
    MemoryDecision,
    MemoryEntry,
    MemoryScope,
    MemoryState,
    MemoryType,
    RetrievalResult,
)
from agent_memory.multiagent import IsolationMode, MultiAgentMemory
from agent_memory.paged_memory import PagedContext, PagedMemory
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.retriever import FusionStrategy, LinearFusionStrategy, RRFFusionStrategy
from agent_memory.scoped import MemoryView
from agent_memory.sqlite_store import SqliteMemoryStore
from agent_memory.vector_index import IvfFlatConfig, VectorIndexConfig

try:
    from agent_memory._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "BackendContext",
    "IvfFlatConfig",
    "VectorIndexConfig",
    "available_backends",
    "register_backend",
    "BenchmarkDataset",
    "BenchmarkHarness",
    "BenchmarkResult",
    "HarnessConfig",
    "HarnessResult",
    "EntityExtractor",
    "EntityType",
    "ExtractedMemory",
    "AgentMemoryError",
    "BackendConnectionError",
    "CompositeExtractionStrategy",
    "ConfigurationError",
    "DecisionError",
    "ExtractionError",
    "FusionStrategy",
    "LinearFusionStrategy",
    "MemoryNotFoundError",
    "MemoryStoreError",
    "MemoryWriteError",
    "RRFFusionStrategy",
    "RetrievalError",
    "configure_debug_logging",
    "get_logger",
    "EntityExtractionStrategy",
    "ExtractionResult",
    "GraphBuilder",
    "KnowledgeEntity",
    "KnowledgeGraph",
    "KnowledgeRelation",
    "PagedContext",
    "PagedMemory",
    "PatternExtractionStrategy",
    "RelationType",
    "SpacyExtractionStrategy",
    "ConfidenceEvent",
    "ConfidenceLearner",
    "ConfidenceUpdate",
    "DecisionEngine",
    "DecisionPolicy",
    "DefaultPolicy",
    "EvalDataset",
    "EvalResult",
    "IsolationMode",
    "Memory",
    "MemoryAction",
    "MemoryDecision",
    "MemoryEdge",
    "MemoryEntry",
    "MemoryGraph",
    "MemoryManager",
    "MemoryScope",
    "MemoryState",
    "MemoryType",
    "MemoryView",
    "MultiAgentMemory",
    "RetrievalResult",
    "SqliteMemoryStore",
    "format_benchmark_report",
    "format_eval_report",
    "run_benchmark",
    "run_eval",
    "run_eval_suite",
]
