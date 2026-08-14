from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote, urlparse

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from runtime_common.common import read_json, relative_to_project, resolve_path


def load_server_config(path: str | Path = "configs/tool_server_config.yaml") -> Dict[str, Any]:
    config_path = resolve_path(path)
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    result: Dict[str, Any] = {}
    current_section: Dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = value
                current_section = None
            else:
                result[key] = {}
                current_section = result[key]
            continue
        if current_section is None:
            continue
        stripped = line.strip()
        key, _, value = stripped.partition(":")
        parsed: Any = value.strip()
        if parsed.isdigit():
            parsed = int(parsed)
        current_section[key.strip()] = parsed
    return result


def build_service_base_url(config: Dict[str, Any]) -> str:
    server = config.get("server", {})
    host = server.get("public_host") or server.get("host") or "127.0.0.1"
    port = server.get("port", 18088)
    return f"http://{host}:{port}"


def build_base_url(config: Dict[str, Any]) -> str:
    server = config.get("server", {})
    browser_base_url = str(server.get("browser_base_url") or "").strip()
    if browser_base_url:
        return browser_base_url.rstrip("/")
    return build_service_base_url(config)


def project_identity(config: Dict[str, Any]) -> Dict[str, str]:
    return {
        "project": "ChronicCare-Agent",
        "base_url": build_base_url(config),
        "service_base_url": build_service_base_url(config),
    }


def safety_note(config: Dict[str, Any]) -> str:
    return config["safety"]["medical_safety_note"]


def artifact_status(path_str: str | Path) -> Dict[str, Any]:
    path = resolve_path(path_str)
    item: Dict[str, Any] = {
        "exists": path.exists(),
        "path": relative_to_project(path),
    }
    if path.exists() and path.is_file():
        item["size_bytes"] = path.stat().st_size
    return item


def read_optional_json(path: str | Path) -> Dict[str, Any]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return {}
    return read_json(file_path)


def load_current_metrics() -> Dict[str, Any]:
    for candidate in [
        "configs/current_metrics.json",
        "outputs/enhanced/current_metrics_snapshot.json",
    ]:
        data = read_optional_json(candidate)
        if data:
            return data
    return {}


def sqlite_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    cfg = load_server_config()
    target = resolve_path(db_path or cfg["paths"]["sqlite_db"])
    return sqlite3.connect(target)


def fetch_rows(sql: str, params: Iterable[Any] | None = None) -> List[Dict[str, Any]]:
    with sqlite_connection() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(sql, tuple(params or ())).fetchall()
    return [dict(row) for row in rows]


def fetch_one(sql: str, params: Iterable[Any] | None = None) -> Dict[str, Any]:
    rows = fetch_rows(sql, params)
    return rows[0] if rows else {}


def ensure_parent(path: str | Path) -> Path:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def public_artifact_url(config: Dict[str, Any], route_path: str) -> str:
    return f"{build_base_url(config)}{route_path}"


def service_artifact_url(config: Dict[str, Any], route_path: str) -> str:
    return f"{build_service_base_url(config)}{route_path}"


def artifact_route_path(route_path: str) -> str:
    route = str(route_path or "").strip()
    if not route:
        return "/"
    return route if route.startswith("/") else f"/{route}"


def _chart_alias_target(filename: str) -> str:
    def latest_dynamic_target(prefix: str) -> str | None:
        matches = []
        for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
            chart_dir = resolve_path(base_str)
            matches.extend(chart_dir.glob(f"{prefix}_*d.svg"))
        matches = sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)
        return matches[0].name if matches else None

    static_map = {
        "line_followup_trend.png": latest_dynamic_target("line_followup_trend") or "line_followup_trend.svg",
        "pie_risk_distribution.png": latest_dynamic_target("pie_risk_distribution") or "pie_risk_distribution.svg",
        "followup_trend_line.png": latest_dynamic_target("line_followup_trend") or "line_followup_trend.svg",
        "risk_distribution_pie.png": latest_dynamic_target("pie_risk_distribution") or "pie_risk_distribution.svg",
        "line_followup_trend_10d.png": "line_followup_trend_10d.svg",
        "pie_risk_distribution_10d.png": "pie_risk_distribution_10d.svg",
        "followup_trend_line_10d.png": "line_followup_trend_10d.svg",
        "disease_inventory_distribution.png": "disease_inventory_distribution.svg",
        "risk_level_distribution.png": "risk_level_distribution.svg",
        "fasting_glucose_distribution.png": "fasting_glucose_distribution.svg",
        "disease_combination_distribution.png": "disease_combination_distribution.svg",
        "disease_combination_distribution.svg": "disease_combination_distribution.svg",
        "cohort_disease_distribution.png": "cohort_disease_distribution_30d.svg",
        "cohort_disease_distribution.svg": "cohort_disease_distribution_30d.svg",
        "hba1c_trend.png": "analysis_trend_hba1c_abnormal_6m.svg",
        "hba1c_trend.svg": "analysis_trend_hba1c_abnormal_6m.svg",
        "hba1c_abnormal_trend_6m.png": "analysis_trend_hba1c_abnormal_6m.svg",
        "hba1c_abnormal_trend_6m.svg": "analysis_trend_hba1c_abnormal_6m.svg",
        "followup_high_risk_45d.png": "line_followup_trend_high_risk_45d.svg",
        "followup_high_risk_45d.svg": "line_followup_trend_high_risk_45d.svg",
    }
    if filename in static_map:
        return static_map[filename]
    patterns = [
        (r"^line_followup_trend_(\d+)d\.png$", "line_followup_trend_{days}d.svg"),
        (r"^followup_trend_line_(\d+)d\.png$", "line_followup_trend_{days}d.svg"),
        (r"^followup_high_risk_(\d+)d\.(?:png|svg)$", "line_followup_trend_high_risk_{days}d.svg"),
        (r"^cohort_disease_distribution_(\d+)d\.(?:png|svg)$", "cohort_disease_distribution_{days}d.svg"),
        (r"^pie_risk_distribution_(\d+)d\.png$", "pie_risk_distribution_{days}d.svg"),
        (r"^risk_distribution_pie_(\d+)d\.png$", "pie_risk_distribution_{days}d.svg"),
        (r"^hba1c_abnormal_trend_(\d+)m\.(?:png|svg)$", "analysis_trend_hba1c_abnormal_{days}m.svg"),
        (r"^hba1c_trend_(\d+)m\.(?:png|svg)$", "analysis_trend_hba1c_abnormal_{days}m.svg"),
    ]
    for pattern, target in patterns:
        match = re.match(pattern, filename)
        if match:
            return target.format(days=match.group(1))
    if filename.endswith(".png"):
        for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
            if resolve_path(f"{base_str}/{filename}").exists():
                return filename
        svg_candidate = filename[:-4] + ".svg"
        for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
            if resolve_path(f"{base_str}/{svg_candidate}").exists():
                return svg_candidate
    return filename


def _graph_driven_alias_target(analysis_id: str) -> str:
    normalized = str(analysis_id or "").strip()
    if normalized.endswith(".html"):
        normalized = normalized[:-5]
    legacy_followup = re.match(r"^followup_high_risk_(\d+)_days$", normalized)
    if legacy_followup:
        return f"analysis_future_followup_chart_bundle_high_risk_{legacy_followup.group(1)}d_chart"
    compact_followup = re.match(r"^analysis_followup_high_risk_(\d+)d(?:_chart)?$", normalized)
    if compact_followup:
        suffix = "_chart" if normalized.endswith("_chart") else ""
        return f"analysis_future_followup_chart_bundle_high_risk_{compact_followup.group(1)}d{suffix}"
    decoded = unquote(normalized)
    if decoded.startswith("kg_subgraph_"):
        return decoded
    if decoded in {
        "analysis_kg_subgraph_hypertension",
        "analysis_kg_subgraph_hypertension_chart",
        "analysis_kg_subgraph_high_salt_hypertension",
        "analysis_kg_subgraph_high_salt_hypertension_chart",
    }:
        return decoded
    alias_map = {
        "analysis_disease_distribution": "analysis_disease_inventory",
        "analysis_disease_distribution_chart": "analysis_disease_inventory_chart",
        "analysis_disease_distribution_30d_followup": "analysis_disease_inventory",
        "analysis_disease_distribution_30d_followup_chart": "analysis_disease_inventory_chart",
        "analysis_patient_disease_distribution": "analysis_disease_inventory",
        "analysis_patient_disease_distribution_chart": "analysis_disease_inventory_chart",
        "analysis_metric_query": "analysis_metric_diabetes_avg_fpg",
        "analysis_metric_query_chart": "analysis_metric_diabetes_avg_fpg_chart",
        "analysis_cohort_disease": "analysis_future_30d_high_risk_followup_disease_distribution",
        "analysis_cohort_disease_chart": "analysis_future_30d_high_risk_followup_disease_distribution_chart",
        "analysis_datamate_pipeline": "analysis_disease_inventory",
        "analysis_datamate_pipeline_chart": "analysis_disease_inventory_chart",
    }
    return alias_map.get(analysis_id, analysis_id)


def artifact_exists_for_route(route_or_url: str) -> bool:
    raw = str(route_or_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    route = parsed.path if parsed.scheme and parsed.netloc else artifact_route_path(raw)
    if route == "/artifacts/charts":
        return True
    if route == "/artifacts/report" or route == "/artifacts/report.md":
        return True
    if route == "/artifacts/graph.html":
        latest = latest_subgraph_path()
        if latest is not None:
            return latest.exists()
        cfg = load_server_config()
        return resolve_path(cfg["paths"]["graph_html"]).exists()
    if route in {
        "/artifacts/kg_subgraph_hypertension_preview.svg",
        "/artifacts/kg_subgraph_high_salt_hypertension_preview.svg",
    }:
        return True
    generic_subgraph = re.match(r"^/artifacts/(?:analysis_)?kg_subgraph_(.+)\.(?:html|svg|png)$", route)
    if generic_subgraph:
        return bool(unquote(generic_subgraph.group(1)).strip())
    if route.startswith("/artifacts/charts/"):
        filename = route.rsplit("/", 1)[-1]
        legacy_followup = re.match(r"^followup_high_risk_(\d+)_days(?:\.html)?$", filename)
        if legacy_followup:
            days = legacy_followup.group(1)
            for base_str in [
                "outputs/runtime_generated/graph_driven_analysis",
                "outputs/local_runtime/graph_driven_analysis",
                "outputs/graph_driven_analysis",
            ]:
                if resolve_path(f"{base_str}/analysis_future_followup_chart_bundle_high_risk_{days}d_chart.html").exists():
                    return True
        compact_followup = re.match(r"^followup_high_risk_(\d+)d\.png$", filename)
        if compact_followup:
            days = compact_followup.group(1)
            for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
                if resolve_path(f"{base_str}/line_followup_trend_high_risk_{days}d.svg").exists():
                    return True
        compact_followup_svg = re.match(r"^followup_high_risk_(\d+)d\.svg$", filename)
        if compact_followup_svg:
            days = compact_followup_svg.group(1)
            for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
                if resolve_path(f"{base_str}/line_followup_trend_high_risk_{days}d.svg").exists():
                    return True
        legacy_subgraph_chart = re.match(r"^kg_subgraph_(.+)\.(html|svg|png)$", filename)
        if legacy_subgraph_chart:
            topic = unquote(legacy_subgraph_chart.group(1)).strip()
            return bool(topic)
        actual_name = _chart_alias_target(filename)
        actual_names = [actual_name]
        if actual_name.endswith(".png"):
            actual_names.append(f"{actual_name[:-4]}.svg")
        for target_name in actual_names:
            for candidate in [
                f"outputs/runtime_generated/charts/{target_name}",
                f"outputs/local_runtime/charts/{target_name}",
                f"outputs/charts/{target_name}",
                f"outputs/runtime_generated/graph_driven_analysis/{target_name}",
                f"outputs/local_runtime/graph_driven_analysis/{target_name}",
                f"outputs/graph_driven_analysis/{target_name}",
            ]:
                if resolve_path(candidate).exists():
                    return True
        return False
    if route.startswith("/artifacts/subgraphs/"):
        subgraph_id = route.rsplit("/", 1)[-1]
        if subgraph_id.endswith(".html"):
            subgraph_id = subgraph_id[:-5]
        if subgraph_id.endswith(".svg"):
            subgraph_id = subgraph_id[:-4]
        if subgraph_id.endswith(".png"):
            subgraph_id = subgraph_id[:-4]
        if not subgraph_id:
            return False
        if subgraph_id == "subgraph_high_salt_hypertension":
            return True
        return any(
            resolve_path(candidate).exists()
            for candidate in [
                f"outputs/runtime_generated/subgraphs/{subgraph_id}.html",
                f"outputs/runtime_generated/subgraphs/{subgraph_id}.png",
                f"outputs/local_runtime/subgraphs/{subgraph_id}.html",
                f"outputs/local_runtime/subgraphs/{subgraph_id}.png",
                f"outputs/subgraphs/{subgraph_id}.html",
            ]
        )
    if route.startswith("/artifacts/graph-driven/"):
        analysis_id = _graph_driven_alias_target(route.rsplit("/", 1)[-1])
        if not analysis_id:
            return False
        if analysis_id.startswith("kg_subgraph_"):
            topic = analysis_id[len("kg_subgraph_") :].strip()
            return bool(topic)
        if analysis_id in {
            "analysis_kg_subgraph_hypertension",
            "analysis_kg_subgraph_hypertension_chart",
            "analysis_kg_subgraph_high_salt_hypertension",
            "analysis_kg_subgraph_high_salt_hypertension_chart",
        }:
            return True
        for base_str in [
            "outputs/runtime_generated/graph_driven_analysis",
            "outputs/local_runtime/graph_driven_analysis",
            "outputs/graph_driven_analysis",
        ]:
            base_dir = resolve_path(base_str)
            direct_path = base_dir / analysis_id
            if direct_path.exists():
                return True
            for suffix in [".html", ".svg", ".png", ".json", ".csv"]:
                if (base_dir / f"{analysis_id}{suffix}").exists():
                    return True
        return False
    if route.startswith("/artifacts/open-nl2sql/"):
        filename = route.rsplit("/", 1)[-1]
        return resolve_path(f"outputs/open_nl2sql/{filename}").exists()
    return False


def latest_subgraph_path() -> Path | None:
    html_paths = []
    for base_str in ["outputs/runtime_generated/subgraphs", "outputs/local_runtime/subgraphs", "outputs/subgraphs"]:
        subgraph_dir = resolve_path(base_str)
        html_paths.extend(subgraph_dir.glob("*.html"))
    html_paths = sorted(html_paths, key=lambda item: item.stat().st_mtime, reverse=True)
    return html_paths[0] if html_paths else None


def latest_subgraph_public_url(config: Dict[str, Any]) -> str:
    latest = latest_subgraph_path()
    if latest is None:
        return public_artifact_url(config, "/artifacts/graph.html")
    version = int(latest.stat().st_mtime)
    return public_artifact_url(config, f"/artifacts/subgraphs/{latest.stem}?v={version}")


def latest_subgraph_service_url(config: Dict[str, Any]) -> str:
    latest = latest_subgraph_path()
    if latest is None:
        return service_artifact_url(config, "/artifacts/graph.html")
    version = int(latest.stat().st_mtime)
    return service_artifact_url(config, f"/artifacts/subgraphs/{latest.stem}?v={version}")


def summarize_metric_rows(rows: List[Dict[str, Any]], key: str, value: str, limit: int = 10) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: item.get(value, 0), reverse=True)
    return [{key: item.get(key), value: item.get(value)} for item in ordered[:limit]]


def file_response_meta(path_str: str | Path) -> Dict[str, Any]:
    target = resolve_path(path_str)
    return {
        "exists": target.exists(),
        "path": relative_to_project(target),
        "size_bytes": target.stat().st_size if target.exists() and target.is_file() else 0,
    }
