"""Small DataMate HTTP client used by task 1."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "task1_datamate.yaml"

# Fallback used when the config file is missing or PyYAML is unavailable. The
# config file mirrors these values and is the authoritative source at runtime.
_DEFAULT_OPERATOR_KEYWORDS = {
    "drop_duplicate_rows": ["duplicate", "dedup"],
    "fill_missing_values": ["missing", "empty"],
    "normalize_column_types": ["text"],
}


def load_operator_keywords(config_path: str | Path | None = None) -> dict[str, list[str]]:
    """Load the DataMate operator keyword mapping from the task-1 config.

    Reads ``datamate.operator_keyword_mapping`` from ``configs/task1_datamate.yaml``
    so the keyword matching is config-driven. Falls back to
    :data:`_DEFAULT_OPERATOR_KEYWORDS` when the file or PyYAML is unavailable.
    """

    path = Path(config_path) if config_path else _CONFIG_PATH
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapping = (data.get("datamate") or {}).get("operator_keyword_mapping")
        if isinstance(mapping, dict) and mapping:
            return {str(op): list(words or []) for op, words in mapping.items()}
    except Exception:
        logger.debug("Falling back to default DataMate operator keywords.", exc_info=True)
    return {op: list(words) for op, words in _DEFAULT_OPERATOR_KEYWORDS.items()}


OPERATOR_KEYWORDS = load_operator_keywords()

OPERATOR_SELECTIONS = {
    "drop_duplicate_rows": [
        "DuplicateFilesFilter",
        "DuplicateSentencesFilter",
        "document_deduplicator",
        "document_minhash_deduplicator",
        "document_simhash_deduplicator",
    ],
    "normalize_column_types": [
        "UnicodeSpaceCleaner",
        "ExtraSpaceCleaner",
        "FullWidthCharacterCleaner",
        "whitespace_normalization_mapper",
        "text_type_normalizer",
    ],
}

LOCAL_ONLY_OPERATORS = {
    "fill_missing_values": (
        "DataMate's current cleaning catalog has no table field imputation "
        "operator. Task 1 handles this as a local preprocessing step before "
        "DataMate text cleaning."
    ),
}


def summarize_hybrid_execution_plan(local_operators: list[str]) -> dict[str, Any]:
    """Describe which operators run locally vs on DataMate for defense evidence."""

    local_pre = [op for op in local_operators if op in LOCAL_ONLY_OPERATORS]
    datamate_ops = [op for op in local_operators if op not in LOCAL_ONLY_OPERATORS]
    return {
        "local_preprocessing": local_pre,
        "local_only_reasons": {
            operator: LOCAL_ONLY_OPERATORS[operator] for operator in local_pre
        },
        "datamate_template_operators": datamate_ops,
        "execution_order": "local_preprocessing -> datamate_template -> persist",
        "hybrid": bool(local_pre and datamate_ops),
    }

TEMPLATE_NAME = "task1-data-cleaning-template"
TEMPLATE_DESCRIPTION = "DataMate cleaning template generated from task 1 local plan."

_ALLOWED_DATAMATE_MODES = frozenset({"dry_run", "submit", "auto"})


def resolve_datamate_mode(
    base_url: str | None,
    mode: str = "dry_run",
    timeout: float = 3.0,
) -> tuple[str, dict[str, Any]]:
    """Resolve ``auto`` to ``submit`` when DataMate is healthy, else ``dry_run``."""

    requested = (mode or "dry_run").lower()
    if requested not in _ALLOWED_DATAMATE_MODES:
        raise ValueError("DataMate mode must be 'dry_run', 'submit', or 'auto'.")

    meta: dict[str, Any] = {
        "requested_mode": requested,
        "resolved_mode": requested,
        "auto_selected": False,
    }
    if requested != "auto":
        return requested, meta

    if not base_url:
        meta["resolved_mode"] = "dry_run"
        meta["reason"] = "no_base_url"
        return "dry_run", meta

    try:
        client = DataMateClient(base_url, timeout=timeout)
        health = client.health()
    except Exception as exc:
        meta["resolved_mode"] = "dry_run"
        meta["reason"] = str(exc) or type(exc).__name__
        return "dry_run", meta

    if health.get("status") == "healthy":
        meta["resolved_mode"] = "submit"
        meta["auto_selected"] = True
        return "submit", meta

    meta["resolved_mode"] = "dry_run"
    meta["reason"] = health.get("status", "unhealthy")
    return "dry_run", meta


class DataMateClient:
    """Minimal stdlib client for the deployed DataMate Python backend."""

    def __init__(self, base_url: str, timeout: float = 3.0) -> None:
        self.base_url = _validate_base_url(base_url)
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/api/health")
        data = payload.get("data") if isinstance(payload, dict) else {}
        service_status = data.get("status") if isinstance(data, dict) else None
        return {
            "status": "healthy" if service_status == "healthy" else "unknown",
            "endpoint": f"{self.base_url}/api/health",
            "service": data.get("service") if isinstance(data, dict) else None,
            "version": data.get("version") if isinstance(data, dict) else None,
            "raw_code": payload.get("code") if isinstance(payload, dict) else None,
        }

    def list_operators(
        self,
        keyword: str | None = None,
        page: int = 0,
        size: int = 10,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"page": page, "size": size}
        if keyword:
            body["keyword"] = keyword
        return self._request_json("POST", "/api/operators/list", body)

    def list_cleaning_templates(
        self,
        keyword: str | None = None,
        page: int = 0,
        size: int = 20,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"page": page, "size": size}
        if keyword:
            query["keyword"] = keyword
        return self._request_json(
            "GET",
            _path_with_query("/api/cleaning/templates", query),
        )

    def get_cleaning_template(self, template_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/api/cleaning/templates/{template_id}")

    def list_cleaning_tasks(
        self,
        status: str | None = None,
        keyword: str | None = None,
        page: int = 0,
        size: int = 10,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"page": page, "size": size}
        if status:
            query["status"] = status
        if keyword:
            query["keyword"] = keyword
        return self._request_json(
            "GET",
            _path_with_query("/api/cleaning/tasks", query),
        )

    def get_cleaning_task(self, task_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/api/cleaning/tasks/{task_id}")

    def catalog_summary(
        self,
        plan_operators: list[str],
        src_dataset_id: str | None = None,
        src_dataset_name: str | None = None,
        dest_dataset_name: str | None = None,
        mode: str = "dry_run",
    ) -> dict[str, Any]:
        first_page = self.list_operators(size=500)
        mappings: dict[str, Any] = {}
        operator_catalog = _operator_index(_payload_data(first_page).get("content", []))
        all_operator_ids = list(operator_catalog.keys())

        for operator in plan_operators:
            if operator in LOCAL_ONLY_OPERATORS:
                mappings[operator] = {
                    "support_level": "local_only",
                    "execution_scope": "local",
                    "keywords": OPERATOR_KEYWORDS.get(operator, []),
                    "match_count": 0,
                    "query_total": 0,
                    "candidate_operator_ids": [],
                    "selected_operator_ids": [],
                    "note": LOCAL_ONLY_OPERATORS[operator],
                }
                continue

            keywords = OPERATOR_KEYWORDS.get(operator, [])
            if not keywords:
                continue
            matches = []
            query_total = 0
            for keyword in keywords:
                payload = self.list_operators(keyword=keyword, size=5)
                data = _payload_data(payload)
                operator_catalog.update(_operator_index(data.get("content", [])))
                query_total += int(data.get("totalElements", 0))
                matches.extend(_operator_ids(data.get("content", [])))
            candidate_ids = sorted(set(matches))
            mappings[operator] = {
                "support_level": "datamate",
                "execution_scope": "datamate",
                "keywords": keywords,
                "match_count": len(candidate_ids),
                "query_total": query_total,
                "candidate_operator_ids": candidate_ids,
                "selected_operator_ids": _select_operator_ids(
                    operator,
                    all_operator_ids,
                    candidate_ids,
                ),
            }

        data = _payload_data(first_page)
        template_payload = self.build_cleaning_template_payload(
            mappings,
            operator_catalog=operator_catalog,
        )
        resolved_dest_dataset_name = dest_dataset_name
        if not resolved_dest_dataset_name and src_dataset_id:
            base_name = f"{src_dataset_name or src_dataset_id}_cleaned"
            if mode == "submit":
                suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
                resolved_dest_dataset_name = f"{base_name}_{suffix}"
            else:
                resolved_dest_dataset_name = base_name
        template_submission = _skipped_submission(
            "/api/cleaning/templates",
            "No DataMate-compatible operators were selected.",
        )
        if template_payload["instance"]:
            template_submission = self.submit_cleaning_template(
                template_payload,
                mode=mode,
            )
        return {
            "status": "available",
            "operator_count": data.get("totalElements", 0),
            "sample_operator_ids": all_operator_ids[:10],
            "candidate_mappings": mappings,
            "cleaning_template": _template_artifact(
                template_payload,
                mappings,
                template_submission,
            ),
            "cleaning_task": _task_artifact(
                template_payload,
                src_dataset_id=src_dataset_id,
                src_dataset_name=src_dataset_name,
                dest_dataset_name=resolved_dest_dataset_name,
                build_payload=self.build_cleaning_task_payload,
                submit_payload=lambda payload: self.submit_cleaning_task(
                    payload,
                    mode=mode,
                ),
            ),
        }

    def build_cleaning_template_payload(
        self,
        candidate_mappings: dict[str, Any],
        operator_catalog: dict[str, dict[str, Any]] | None = None,
        name: str = TEMPLATE_NAME,
        description: str = TEMPLATE_DESCRIPTION,
    ) -> dict[str, Any]:
        operator_catalog = operator_catalog or {}
        selected_ids: list[str] = []
        for mapping in candidate_mappings.values():
            for operator_id in mapping.get("selected_operator_ids", []):
                if operator_id not in selected_ids:
                    selected_ids.append(operator_id)

        return {
            "name": name,
            "description": description,
            "instance": [
                _operator_template_instance(
                    operator_id,
                    operator_catalog.get(operator_id, {}),
                )
                for operator_id in selected_ids
            ],
        }

    def build_cleaning_task_payload(
        self,
        template_payload: dict[str, Any],
        src_dataset_id: str,
        src_dataset_name: str,
        dest_dataset_name: str,
        dest_dataset_type: str = "TEXT",
    ) -> dict[str, Any]:
        return {
            "name": f"{src_dataset_name}-task1-cleaning",
            "description": "DataMate cleaning task generated from task 1 plan.",
            "srcDatasetId": src_dataset_id,
            "srcDatasetName": src_dataset_name,
            "destDatasetName": dest_dataset_name,
            "destDatasetType": dest_dataset_type,
            "instance": template_payload.get("instance", []),
        }

    def create_cleaning_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", "/api/cleaning/templates", payload)

    def create_cleaning_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", "/api/cleaning/tasks", payload)

    def submit_cleaning_template(
        self,
        payload: dict[str, Any],
        mode: str = "dry_run",
    ) -> dict[str, Any]:
        submission_payload = _with_unique_submission_name(payload, mode)
        return self._submit_payload(
            endpoint="/api/cleaning/templates",
            payload=submission_payload,
            submit=lambda: self.create_cleaning_template(submission_payload),
            verify=self.get_cleaning_template,
            mode=mode,
        )

    def submit_cleaning_task(
        self,
        payload: dict[str, Any],
        mode: str = "dry_run",
    ) -> dict[str, Any]:
        submission_payload = _with_unique_submission_name(payload, mode)
        return self._submit_payload(
            endpoint="/api/cleaning/tasks",
            payload=submission_payload,
            submit=lambda: self.create_cleaning_task(submission_payload),
            verify=self.get_cleaning_task,
            mode=mode,
        )

    def _submit_payload(
        self,
        endpoint: str,
        payload: dict[str, Any],
        submit: Callable[[], dict[str, Any]],
        mode: str,
        verify: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if mode not in {"dry_run", "submit"}:
            return {
                "mode": mode,
                "submitted": False,
                "status": "invalid_mode",
                "endpoint": endpoint,
                "message": "DataMate mode must be 'dry_run' or 'submit'.",
            }
        if mode == "dry_run":
            return {
                "mode": "dry_run",
                "submitted": False,
                "status": "prepared",
                "endpoint": endpoint,
                "payload": payload,
            }

        try:
            response = submit()
        except Exception as exc:  # Keep submit failures visible in run artifacts.
            return _submission_failure(endpoint, exc)
        result = {
            "mode": "submit",
            "submitted": True,
            "status": "submitted",
            "endpoint": endpoint,
            "response": response,
        }
        resource_id = _response_resource_id(response)
        if not resource_id:
            return result

        result["resource_id"] = resource_id
        if verify is None:
            return result
        try:
            verification = verify(resource_id)
        except Exception as exc:
            result["status"] = "submitted_unverified"
            result["verification_error"] = _submission_failure(endpoint, exc)
            return result

        verified_id = _response_resource_id(verification)
        result["verification"] = verification
        result["verified"] = verified_id == resource_id
        result["status"] = "verified" if result["verified"] else "submitted_unverified"
        return result

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self.base_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def safe_datamate_call(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run a DataMate call and normalize network/JSON failures."""

    try:
        return operation()
    except HTTPError as exc:
        error = _http_error_details(exc)
        return {
            "status": "unavailable",
            "message": error["message"],
            "http_status": error["http_status"],
            "body": error["body"],
        }
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "message": str(exc)}


def _validate_base_url(base_url: str) -> str:
    parsed = urlparse((base_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("DataMate base_url must be an absolute http(s) URL.")
    if parsed.username or parsed.password:
        raise ValueError("DataMate base_url must not include credentials.")
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def _payload_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _path_with_query(path: str, query: dict[str, Any]) -> str:
    return f"{path}?{urlencode(query)}"


def _response_resource_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and data.get("id"):
        return str(data["id"])
    if isinstance(payload, dict) and payload.get("id"):
        return str(payload["id"])
    return None


def _operator_ids(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        str(item["id"])
        for item in content
        if isinstance(item, dict) and item.get("id")
    ]


def _operator_index(content: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(content, list):
        return {}
    return {
        str(item["id"]): item
        for item in content
        if isinstance(item, dict) and item.get("id")
    }


def _operator_template_instance(
    operator_id: str,
    operator_record: dict[str, Any],
) -> dict[str, Any]:
    instance: dict[str, Any] = {"id": operator_id}
    for field in ("name", "description", "inputs", "outputs", "categories", "settings"):
        value = operator_record.get(field)
        if value is not None:
            instance[field] = value
    instance["overrides"] = operator_record.get("overrides") or {}
    return instance


def _select_operator_ids(
    local_operator: str,
    all_operator_ids: list[str],
    candidate_operator_ids: list[str],
) -> list[str]:
    available = set(all_operator_ids) | set(candidate_operator_ids)
    selected = [
        operator_id
        for operator_id in OPERATOR_SELECTIONS.get(local_operator, [])
        if operator_id in available
    ]
    if selected:
        return selected[:2]
    return candidate_operator_ids[:2]


def _template_artifact(
    template_payload: dict[str, Any],
    candidate_mappings: dict[str, Any],
    submission: dict[str, Any],
) -> dict[str, Any]:
    local_only = [
        operator
        for operator, mapping in candidate_mappings.items()
        if mapping.get("execution_scope") == "local"
    ]
    if not template_payload["instance"]:
        return {
            "status": "skipped",
            "endpoint": "/api/cleaning/templates",
            "payload": template_payload,
            "submission": submission,
            "local_only_operators": local_only,
            "message": "No DataMate-compatible operators were selected.",
        }
    return {
        "status": "ready",
        "endpoint": "/api/cleaning/templates",
        "payload": template_payload,
        "submission": submission,
        "local_only_operators": local_only,
        "message": (
            "Payload is ready for POST /api/cleaning/templates; submission is "
            "kept explicit because it mutates the DataMate database."
        ),
    }


def _task_artifact(
    template_payload: dict[str, Any],
    src_dataset_id: str | None,
    src_dataset_name: str | None,
    dest_dataset_name: str | None,
    build_payload: Callable[..., dict[str, Any]],
    submit_payload: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if not src_dataset_id:
        return {
            "status": "waiting_for_dataset",
            "endpoint": "/api/cleaning/tasks",
            "required_fields": ["srcDatasetId", "srcDatasetName"],
            "message": (
                "Provide an existing DataMate dataset id to build or submit "
                "a cleaning task."
            ),
        }
    if not template_payload["instance"]:
        return {
            "status": "skipped",
            "endpoint": "/api/cleaning/tasks",
            "payload": None,
            "submission": _skipped_submission(
                "/api/cleaning/tasks",
                "No DataMate-compatible operators were selected.",
            ),
            "message": "No DataMate cleaning task can be built without template instances.",
        }

    resolved_src_name = src_dataset_name or src_dataset_id
    payload = build_payload(
        template_payload=template_payload,
        src_dataset_id=src_dataset_id,
        src_dataset_name=resolved_src_name,
        dest_dataset_name=dest_dataset_name or f"{resolved_src_name}_cleaned",
    )
    return {
        "status": "ready",
        "endpoint": "/api/cleaning/tasks",
        "payload": payload,
        "submission": submit_payload(payload),
        "message": (
            "Payload is ready for POST /api/cleaning/tasks; creation will "
            "execute the DataMate cleaning task."
        ),
    }


def _skipped_submission(endpoint: str, message: str) -> dict[str, Any]:
    return {
        "mode": "skipped",
        "submitted": False,
        "status": "skipped",
        "endpoint": endpoint,
        "message": message,
    }


def _with_unique_submission_name(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode != "submit" or not payload.get("name"):
        return payload
    unique_payload = {**payload}
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    unique_payload["name"] = f"{payload['name']}-{suffix}"
    return unique_payload


def _submission_failure(endpoint: str, exc: Exception) -> dict[str, Any]:
    failure = {
        "mode": "submit",
        "submitted": False,
        "status": "submit_failed",
        "endpoint": endpoint,
        "message": str(exc),
    }
    if isinstance(exc, HTTPError):
        error = _http_error_details(exc)
        failure["message"] = error["message"]
        failure["http_status"] = error["http_status"]
        failure["body"] = error["body"]
    return failure


def _http_error_details(exc: HTTPError) -> dict[str, Any]:
    body = ""
    try:
        body = exc.read().decode("utf-8")
    except Exception:
        body = ""
    message = f"HTTP {exc.code} {exc.reason}"
    if body:
        message = f"{message}: {body}"
    return {
        "http_status": exc.code,
        "body": body,
        "message": message,
    }
