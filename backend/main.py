"""FastAPI async job dispatch (Section 2.1 top box).

Endpoints:
  POST /research      -> submit a query, get back a job_id immediately
  GET  /status/{id}   -> poll current status + budget usage
  GET  /result/{id}   -> get the final answer once the job's done

Job lifecycle:
  1. new_job_state() snapshots hyde_enabled, sf_threshold, and the two
     budget caps (Section 1.3 #6). These are frozen for this job — mid-
     execution slider changes on the UI cannot corrupt a running job.
  2. HyDEOperator.execute() runs once, outside the graph (Section 2.1's
     "Pre-flight Layer (OUTSIDE state machine)"). Its call counts against
     the LLM budget even though it happens before the graph starts.
  3. The compiled LangGraph runs asynchronously; the request handler
     doesn't wait for it. Results are polled via /status and /result.

Storage: in-memory dict keyed by job_id. Section 8's stated scope is
"single-user demo, concurrent multi-user load not supported" — a real
service would swap this for Redis/Postgres. Deliberately not doing that
here because it would be scope creep for a portfolio demo and would
obscure the interesting parts (the multi-agent orchestration, not the
job store).
"""
import asyncio
import logging
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from backend.graph import compiled_graph
from backend.nodes.preflight import HyDEOperator
from backend.state import AcademicResearchState, new_job_state

log = logging.getLogger(__name__)

app = FastAPI(title="Academic Research Agent")

_jobs: dict[str, AcademicResearchState] = {}
_graph = None
_hyde: HyDEOperator | None = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = compiled_graph()
    return _graph


def _get_hyde() -> HyDEOperator:
    global _hyde
    if _hyde is None:
        _hyde = HyDEOperator()
    return _hyde


class ResearchRequest(BaseModel):
    query: str
    hyde_enabled: bool = True
    sf_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    max_critic_loops: int = Field(default=3, ge=1, le=10)
    max_llm_calls: int = Field(default=15, ge=1, le=100)


class ResearchResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    total_llm_calls: int
    max_llm_calls: int
    llm_budget_exceeded: bool
    critic_loop_count: int
    coverage_score: float
    llm_calls_avoided: int


class ResultResponse(BaseModel):
    job_id: str
    status: str
    final_answer: str
    citations: list[str]
    failure_reason: Optional[str]
    total_llm_calls: int
    llm_calls_avoided: int
    coverage_score: float
    cascade_decisions: list[str]
    gamma_score_mean: Optional[float]


async def _run_job(state: AcademicResearchState) -> None:
    job_id = state["job_id"]
    try:
        # ── Pre-flight (Section 2.1): HyDE runs once, OUTSIDE the graph.
        search_payload, hyde_used = await _get_hyde().execute(
            state["query"], state["hyde_enabled"]
        )
        state["search_payload"] = search_payload
        if hyde_used:
            new_total = state["total_llm_calls"] + 1
            state["total_llm_calls"] = new_total
            state["llm_budget_exceeded"] = new_total >= state["max_llm_calls"]

        final = await _get_graph().ainvoke(state)
        _jobs[job_id] = final
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        state["status"] = "failed"
        state["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _jobs[job_id] = state


@app.post("/research", response_model=ResearchResponse)
async def submit_research(request: ResearchRequest) -> ResearchResponse:
    job_id = str(uuid.uuid4())
    state = new_job_state(
        job_id=job_id,
        query=request.query,
        hyde_enabled=request.hyde_enabled,
        sf_threshold=request.sf_threshold,
        max_critic_loops=request.max_critic_loops,
        max_llm_calls=request.max_llm_calls,
    )
    _jobs[job_id] = state
    asyncio.create_task(_run_job(state))
    return ResearchResponse(job_id=job_id)


def _require_job(job_id: str) -> AcademicResearchState:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return _jobs[job_id]


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    state = _require_job(job_id)
    return StatusResponse(
        job_id=job_id,
        status=state["status"],
        total_llm_calls=state["total_llm_calls"],
        max_llm_calls=state["max_llm_calls"],
        llm_budget_exceeded=state["llm_budget_exceeded"],
        critic_loop_count=state["critic_loop_count"],
        coverage_score=state["coverage_score"],
        llm_calls_avoided=state["llm_calls_avoided"],
    )


TERMINAL_STATUSES = {"approved", "failed", "force_finalized", "budget_exceeded"}


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str) -> ResultResponse:
    state = _require_job(job_id)
    if state["status"] not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409, detail=f"Job not finished; current status={state['status']!r}"
        )
    scores = state.get("gamma_scores") or []
    gamma_mean = float(sum(scores) / len(scores)) if scores else None
    return ResultResponse(
        job_id=job_id,
        status=state["status"],
        final_answer=state["final_answer"],
        citations=state["citations"],
        failure_reason=state["failure_reason"],
        total_llm_calls=state["total_llm_calls"],
        llm_calls_avoided=state["llm_calls_avoided"],
        coverage_score=state["coverage_score"],
        cascade_decisions=state["cascade_decisions"],
        gamma_score_mean=gamma_mean,
    )
