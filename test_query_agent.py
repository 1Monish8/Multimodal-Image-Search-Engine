import json

from src.query_agent import TraceLogger, parse_query


def test_parse_query_builds_structured_negative_plan():
    plan = parse_query("Find red sports cars without vintage styling", top_k=100)

    assert plan.query == "Find red sports cars"
    assert plan.negative_prompt == "vintage styling"
    assert plan.top_k == 50
    assert plan.rationale


def test_trace_logger_writes_jsonl_without_payload_bytes(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    TraceLogger(trace_path).write("tool_completed", request_id="abc", result_count=2)

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["event"] == "tool_completed"
    assert record["request_id"] == "abc"
    assert "image_bytes" not in record