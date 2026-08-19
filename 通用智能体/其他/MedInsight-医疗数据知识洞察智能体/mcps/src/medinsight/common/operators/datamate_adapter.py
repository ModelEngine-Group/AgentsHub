from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict

from medinsight.common.operators.base import Operator


class DataMateAdapter:
    """Thin adapter for DataMate discovery and task-facing operators."""

    def __init__(
        self,
        enabled: bool | None = None,
        base_url: str | None = None,
        management_base_url: str | None = None,
        timeout: float = 2.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("DATAMATE_BASE_URL") or "http://localhost:18000").rstrip("/")
        self.management_base_url = (
            management_base_url or os.getenv("DATAMATE_MANAGEMENT_BASE_URL") or "http://localhost:8080"
        ).rstrip("/")
        self.enabled = bool(enabled) if enabled is not None else os.getenv("DATAMATE_ENABLED", "auto") != "false"
        self.timeout = timeout
        self.last_context: Dict[str, Any] | None = None

    def run(self, operator_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.enabled:
            raise NotImplementedError(
                "Direct DataMate operator execution is not configured; use execute() so runtime status is traced."
            )
        return {"operator": operator_name, "payload": payload, "runtime": "local_operator", "fallback_used": True}

    def execute(self, operator: Operator, payload: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        fallback = (self.last_context or {}).get("fallback") or {}
        fallback_used = bool(fallback.get("used", True))
        try:
            result = operator.run(payload)
            status = "done"
            error = None
        except Exception as exc:
            result = {}
            status = "failed"
            error = str(exc)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        trace = {
            "operator": operator.spec.name,
            "runtime": "datamate_verified_local_operator" if not fallback_used else "local_operator_fallback",
            "datamate_adapter_enabled": self.enabled,
            "datamate_connected": bool((self.last_context or {}).get("connected")),
            "fallback_used": fallback_used,
            "fallback_reason": fallback.get("reason"),
            "missing_services": fallback.get("missing_services", []),
            "status": status,
            "duration_ms": duration_ms,
        }
        if error:
            trace["error"] = error
            raise RuntimeError(trace) from None
        result["_trace"] = trace
        return result

    def discover_capabilities(self) -> Dict[str, Any]:
        started = time.perf_counter()
        context: Dict[str, Any] = {
            "enabled": self.enabled,
            "base_url": self.base_url,
            "management_base_url": self.management_base_url,
            "connected": False,
            "runtime": "datamate-python",
            "health": None,
            "cleaning_templates": [],
            "operator_catalog": {},
            "service_checks": [],
            "fallback": {
                "used": False,
                "reason": None,
                "missing_services": [],
            },
            "error": None,
        }
        if not self.enabled:
            context["error"] = "DataMate integration disabled by DATAMATE_ENABLED=false."
            context["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self._set_fallback(
                context,
                "DataMate integration is disabled by DATAMATE_ENABLED=false.",
                ["datamate_python_backend", "datamate_gateway"],
            )
            self.last_context = context
            return context

        health = self._checked_call(context, "datamate_python_health", f"{self.base_url}/api/health", self._get_json, "/api/health")
        templates = self._checked_call(
            context,
            "datamate_cleaning_templates",
            f"{self.base_url}/api/cleaning/templates",
            self._get_json,
            "/api/cleaning/templates",
        )
        operators = self._checked_call(
            context,
            "datamate_operator_catalog",
            f"{self.base_url}/api/operators/list",
            self._post_json,
            "/api/operators/list",
            {"page": 0, "size": 10},
        )
        self._checked_reachable(context, "datamate_gateway", self.management_base_url)

        if isinstance(health, dict) and health.get("code") == "0" and isinstance(templates, dict) and isinstance(operators, dict):
            context.update(
                {
                    "connected": True,
                    "health": health.get("data"),
                    "cleaning_templates": self._summarize_templates(templates),
                    "operator_catalog": self._summarize_operators(operators),
                }
            )
        else:
            failed_services = [
                item["name"]
                for item in context["service_checks"]
                if item.get("required") and item.get("status") != "ok"
            ]
            self._set_fallback(
                context,
                "DataMate required service/API is unavailable; Task1 used local task operators.",
                failed_services,
            )
            context["error"] = context["fallback"]["reason"]
        context["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        self.last_context = context
        return context

    def _set_fallback(self, context: Dict[str, Any], reason: str, missing_services: list[str]) -> None:
        context["fallback"] = {
            "used": True,
            "reason": reason,
            "missing_services": missing_services,
            "action": "Task1 kept the run reproducible with local task operators; start the missing services to remove fallback.",
        }

    def _checked_call(
        self,
        context: Dict[str, Any],
        name: str,
        url: str,
        func: Any,
        *args: Any,
    ) -> Dict[str, Any] | None:
        check: Dict[str, Any] = {"name": name, "url": url, "required": True, "status": "unknown", "error": None}
        try:
            payload = func(*args)
            check["status"] = "ok"
            return payload
        except Exception as exc:
            check["status"] = "unavailable"
            check["error"] = self._format_error(exc)
            return None
        finally:
            context["service_checks"].append(check)

    def _checked_reachable(self, context: Dict[str, Any], name: str, url: str) -> None:
        check: Dict[str, Any] = {"name": name, "url": url, "required": False, "status": "unknown", "error": None}
        try:
            request = urllib.request.Request(url, headers={"Accept": "text/html,application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout):
                check["status"] = "ok"
        except urllib.error.HTTPError as exc:
            check["status"] = "ok" if exc.code < 500 else "unavailable"
            if exc.code >= 500:
                check["error"] = self._format_error(exc)
        except Exception as exc:
            check["status"] = "unavailable"
            check["error"] = self._format_error(exc)
        context["service_checks"].append(check)

    def _format_error(self, exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            return f"HTTP {exc.code}: {exc.reason}"
        if isinstance(exc, urllib.error.URLError):
            return str(exc.reason)
        return str(exc)

    def run_cleaning_task_probe(self, records: list[dict[str, Any]], wait_seconds: float = 1.0) -> Dict[str, Any]:
        started = time.perf_counter()
        timestamp = int(time.time())
        dataset_name = f"medinsight_probe_{timestamp}"
        file_name = "medinsight_records.jsonl"
        file_bytes = "\n".join(json.dumps(row, ensure_ascii=False) for row in records).encode("utf-8") + b"\n"

        report: Dict[str, Any] = {
            "dataset": None,
            "upload": None,
            "cleaning_task": None,
            "task_detail": None,
            "task_log": None,
            "stopped": False,
            "error": None,
        }
        try:
            dataset = self.create_dataset(dataset_name, "MedInsight DataMate cleaning probe")
            report["dataset"] = dataset
            upload = self.upload_dataset_file(dataset["id"], file_name, file_bytes)
            report["upload"] = upload
            task = self.create_cleaning_task(
                src_dataset_id=dataset["id"],
                src_dataset_name=dataset["name"],
                task_name=f"medinsight_cleaning_probe_{timestamp}",
                dest_dataset_name=f"medinsight_probe_cleaned_{timestamp}",
                operator_id="ExtraSpaceCleaner",
                operator_name="多余空格去除",
            )
            report["cleaning_task"] = task
            time.sleep(wait_seconds)
            task_id = task["id"]
            detail = self._get_json(f"/api/cleaning/tasks/{task_id}")
            report["task_detail"] = detail.get("data")
            log = self._get_json(f"/api/cleaning/tasks/{task_id}/log/0")
            report["task_log"] = log.get("data")
            if (report["task_detail"] or {}).get("status") == "RUNNING":
                self._post_json(f"/api/cleaning/tasks/{task_id}/stop", {})
                report["stopped"] = True
        except Exception as exc:
            report["error"] = str(exc)
        report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return report

    def create_dataset(self, name: str, description: str) -> Dict[str, Any]:
        response = self._post_json(
            "/api/data-management/datasets",
            {
                "name": name,
                "description": description,
                "datasetType": "TEXT",
                "tags": ["medinsight"],
                "dataSource": "LOCAL",
                "status": "ACTIVE",
            },
            base_url=self.management_base_url,
        )
        self._assert_success(response, "create_dataset")
        return response["data"]

    def upload_dataset_file(self, dataset_id: str, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        pre_upload = self._post_json(
            f"/api/data-management/datasets/{dataset_id}/files/upload/pre-upload",
            {"totalFileNum": 1, "totalSize": len(file_bytes), "prefix": ""},
            base_url=self.management_base_url,
        )
        self._assert_success(pre_upload, "pre_upload")
        req_id = pre_upload["data"]
        self._post_multipart(
            f"/api/data-management/datasets/{dataset_id}/files/upload/chunk",
            fields={
                "reqId": req_id,
                "fileNo": "1",
                "fileName": file_name,
                "fileSize": str(len(file_bytes)),
                "totalChunkNum": "1",
                "chunkNo": "1",
            },
            files={"file": (file_name, file_bytes, "application/jsonl")},
            base_url=self.management_base_url,
        )
        files = self._get_json(
            f"/api/data-management/datasets/{dataset_id}/files?page=0&size=10",
            base_url=self.management_base_url,
        )
        self._assert_success(files, "list_dataset_files")
        return {"req_id": req_id, "files": files["data"]}

    def create_cleaning_task(
        self,
        src_dataset_id: str,
        src_dataset_name: str,
        task_name: str,
        dest_dataset_name: str,
        operator_id: str,
        operator_name: str,
    ) -> Dict[str, Any]:
        response = self._post_json(
            "/api/cleaning/tasks",
            {
                "name": task_name,
                "description": "MedInsight probe task created through DataMate cleaning API.",
                "src_dataset_id": src_dataset_id,
                "src_dataset_name": src_dataset_name,
                "dest_dataset_name": dest_dataset_name,
                "dest_dataset_type": "TEXT",
                "instance": [
                    {
                        "id": operator_id,
                        "name": operator_name,
                        "inputs": "text",
                        "outputs": "text",
                        "overrides": {},
                    }
                ],
            },
        )
        self._assert_success(response, "create_cleaning_task")
        return response["data"]

    def _get_json(self, path: str, base_url: str | None = None) -> Dict[str, Any]:
        request = urllib.request.Request(f"{base_url or self.base_url}{path}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: Dict[str, Any], base_url: str | None = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url or self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {"code": "0", "message": "success", "data": None}

    def _post_multipart(
        self,
        path: str,
        fields: Dict[str, str],
        files: Dict[str, tuple[str, bytes, str]],
        base_url: str | None = None,
    ) -> Dict[str, Any]:
        boundary = f"----MedInsightBoundary{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        for name, (filename, content, content_type) in files.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"),
                    f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                    content,
                    b"\r\n",
                ]
            )
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        data = b"".join(chunks)
        request = urllib.request.Request(
            f"{base_url or self.base_url}{path}",
            data=data,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=max(self.timeout, 10)) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {"code": "0", "message": "success", "data": None}

    def _assert_success(self, response: Dict[str, Any], stage: str) -> None:
        if response.get("code") not in {"0", 0, None}:
            raise RuntimeError(f"DataMate {stage} failed: {response}")

    def _summarize_templates(self, payload: Dict[str, Any]) -> list[Dict[str, Any]]:
        data = payload.get("data") or {}
        content = data.get("content") or []
        summaries: list[Dict[str, Any]] = []
        for item in content[:5]:
            instances = item.get("instance") or []
            summaries.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "operator_count": len(instances),
                    "operators": [
                        {
                            "id": operator.get("id"),
                            "name": operator.get("name"),
                            "inputs": operator.get("inputs"),
                            "outputs": operator.get("outputs"),
                        }
                        for operator in instances[:5]
                    ],
                }
            )
        return summaries

    def _summarize_operators(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data") or {}
        content = data.get("content") or []
        text_ops = [item for item in content if item.get("inputs") == "text" or item.get("outputs") == "text"]
        return {
            "total": data.get("totalElements"),
            "sample_size": len(content),
            "text_operator_count_in_sample": len(text_ops),
            "sample_operators": [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "inputs": item.get("inputs"),
                    "outputs": item.get("outputs"),
                }
                for item in content[:5]
            ],
        }
