from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataMatePipelineRunRequest(BaseModel):
    task_id: str = Field(default="datamate_pipeline_api_001", description="Task id for this DataMate pipeline request.")
    force: bool = Field(default=False, description="Whether to rerun even when the latest outputs already exist.")
    safe_run: bool = Field(default=True, description="Whether to use the safe-run workflow with backup and sync.")
    use_npu: bool = Field(default=False, description="Whether to run DataMate NPU-enhanced branches.")
    npu_targets: List[str] = Field(default_factory=list, description="NPU-enhanced operator names to run, e.g. entity/relation BGE enhancement and NL2SQL similarity.")
    fallback: bool = Field(default=True, description="Whether to fall back to CPU artifacts when NPU runtime/model service is unavailable.")


class DataMateDagPlanRequest(BaseModel):
    goal: str = Field(default="full", description="clean, kg, sqlite, analysis, or full")
    input_path: Optional[str] = Field(default=None, description="Optional input path; defaults to data/raw")
    use_npu: bool = Field(default=False)


class DataMateDagRunRequest(DataMateDagPlanRequest):
    dry_run: bool = Field(default=False)
    resume_run_id: Optional[str] = Field(default=None)
    resume_from: Optional[str] = Field(default=None)


class NPUBenchmarkRequest(BaseModel):
    use_npu: bool = Field(default=True, description="Whether to attempt the NPU branch.")
    fallback: bool = Field(default=True, description="Whether to fall back to CPU artifacts when NPU is unavailable.")


class KGQueryRequest(BaseModel):
    query_type: str = Field(description="Supported types include disease_profile, drug_profile, indicator_profile, patient_overview.")
    entity_id: str = Field(description="Canonical entity id such as Disease::hypertension.")


class KGTextQueryRequest(BaseModel):
    query: str = Field(description="Natural-language question for graph query.")
    max_nodes: int = Field(default=80, description="Maximum nodes for subgraph query or render.")


class PatientPathQueryRequest(BaseModel):
    patient_id: str = Field(description="Patient id such as P0001.")
    max_hops: int = Field(default=3, description="Reserved hop count for patient path query.")


class AnalysisQueryRequest(BaseModel):
    question: str = Field(description="Natural-language analysis question for the stable analysis surface.")


class OpenAnalysisQueryRequest(BaseModel):
    question: str = Field(description="Open natural-language analysis question that may need planner routing, synonym rewrite, or fallback.")


class OpenSQLQueryRequest(BaseModel):
    question: str = Field(description="Open chronic-care SQL question.")
    prefer_llm: bool = Field(default=True, description="Whether to allow stage-2 LLM SQL candidate when template stage misses.")
    force_llm: bool = Field(default=False, description="Whether to try stage-2 LLM SQL candidate before template SQL; SQL Guard still applies.")
    allow_chart: bool = Field(default=True, description="Whether to generate chart artifact for trend/distribution results.")
    as_of_date: Optional[str] = Field(default=None, description="Optional YYYY-MM-DD analysis date override; defaults to the current Asia/Shanghai date.")


class AgentGoalRequest(BaseModel):
    user_goal: str = Field(description="User goal to be decomposed into local multi-agent steps.")


class ToolResponse(BaseModel):
    status: str
    safety_note: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
