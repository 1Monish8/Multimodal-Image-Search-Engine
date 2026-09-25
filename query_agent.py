"""Structured query planning and traceable tool execution for multimodal retrieval."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import EVAL_DIR

LOGGER = logging.getLogger(__name__)
TRACE_PATH = EVAL_DIR / "query_traces.jsonl"


@dataclass
class QueryPlan:
    """Validated, serializable representation of a retrieval request."""

    query: str
    mode: str = "text"
    negative_prompt: Optional[str] = None
    top_k: int = 8
    category: Optional[str] = None
    color: Optional[str] = None
    confidence: float = 1.0
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_query(query: str, top_k: int = 8) -> QueryPlan:
    """Convert free text into a deterministic structured retrieval plan.

    This is intentionally local and auditable: it provides the structured-output
    contract an LLM planner could satisfy later without making the demo depend on
    a network service or sending user data to a third party.
    """
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("Query must contain at least one non-whitespace character.")

    plan = QueryPlan(query=normalized, top_k=max(1, min(int(top_k), 50)))
    lower_query = normalized.lower()

    negative_match = re.search(r"\b(?:without|excluding|exclude|not)\s+(.+)$", lower_query)
    if negative_match:
        plan.negative_prompt = normalized[negative_match.start(1):].strip(" .,;:")
        plan.query = normalized[:negative_match.start()].strip(" .,;:") or normalized
        plan.rationale.append("Detected an exclusion clause and selected negative-prompt retrieval.")

    if any(token in lower_query for token in ("image", "photo", "picture", "similar to")):
        plan.rationale.append("Kept text retrieval because no image payload was supplied.")
    if "hybrid" in lower_query:
        plan.mode = "hybrid"
        plan.rationale.append("Detected hybrid intent; an image payload is required to execute it.")

    plan.confidence = 0.9 if plan.rationale else 0.75
    return plan


class TraceLogger:
    """Writes query events without storing image bytes or sensitive payloads."""

    def __init__(self, trace_path: Path = TRACE_PATH):
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, default=str) + "\n")
        LOGGER.info("query_event=%s request_id=%s", event, payload.get("request_id"))
        return record


class RetrievalAgent:
    """Small tool-calling agent over the existing FAISS indexer."""

    def __init__(self, indexer: Any, trace_logger: Optional[TraceLogger] = None):
        self.indexer = indexer
        self.trace_logger = trace_logger or TraceLogger()
        self.tools: Dict[str, Callable[..., Tuple[List[Dict[str, Any]], float]]] = {
            "text_search": indexer.search_by_text,
        }
        negative_search = getattr(indexer, "search_with_negative_prompt", None)
        if callable(negative_search):
            self.tools["negative_text_search"] = negative_search

    def run(self, query: str, top_k: int = 8, request_id: Optional[str] = None) -> Dict[str, Any]:
        request_id = request_id or uuid.uuid4().hex
        plan = parse_query(query, top_k=top_k)
        self.trace_logger.write("plan_created", request_id=request_id, plan=plan.to_dict())

        if plan.mode == "hybrid":
            raise ValueError("Hybrid plans require an image payload; use the existing hybrid search control.")

        started = time.perf_counter()
        if plan.negative_prompt:
            tool_name = "negative_text_search"
            if tool_name not in self.tools:
                raise ValueError("Negative-prompt retrieval is unavailable in the loaded indexer. Restart Streamlit to reload the latest code.")
            results, latency_ms = self.tools[tool_name](
                plan.query,
                plan.negative_prompt,
                beta=0.4,
                top_k=plan.top_k,
            )
        else:
            tool_name = "text_search"
            results, latency_ms = self.tools[tool_name](plan.query, top_k=plan.top_k)

        total_latency_ms = (time.perf_counter() - started) * 1000.0
        trace = self.trace_logger.write(
            "tool_completed",
            request_id=request_id,
            tool=tool_name,
            result_count=len(results),
            retrieval_latency_ms=round(latency_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
        )
        return {
            "request_id": request_id,
            "plan": plan.to_dict(),
            "tool": tool_name,
            "results": results,
            "latency_ms": latency_ms,
            "trace": trace,
        }


def format_structured_result(result: Dict[str, Any]) -> str:
    """Return a compact JSON representation suitable for UI/API output."""
    return json.dumps(
        {
            "request_id": result["request_id"],
            "plan": result["plan"],
            "tool": result["tool"],
            "result_count": len(result["results"]),
            "latency_ms": round(result["latency_ms"], 3),
        },
        indent=2,
    )
