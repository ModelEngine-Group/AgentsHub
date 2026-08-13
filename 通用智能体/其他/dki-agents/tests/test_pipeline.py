"""基础层测试：LocalPipelineRunner 语义（map/filter/失败隔离/状态机/指标）。"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from dki.dmcompat import Fields, Mapper, OPERATORS
from dki.pipeline import LocalPipelineRunner, TaskStatus, make_sample
from dki.pipeline.config import _normalize


def _make_runtime_root(root):
    (root / "ops").mkdir(parents=True)
    (root / "python-executor").mkdir(parents=True)
    return root


def _write_fake_datamate(runtime_root, base_op_source):
    package_root = runtime_root / "python-executor" / "datamate"
    core_root = package_root / "core"
    core_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (core_root / "__init__.py").write_text("", encoding="utf-8")
    (core_root / "base_op.py").write_text(
        textwrap.dedent(base_op_source),
        encoding="utf-8",
    )
    (core_root / "constant.py").write_text(
        "class Fields:\n    result = 'result'\n",
        encoding="utf-8",
    )


def _run_dmcompat_subprocess(runtime_root, code):
    env = os.environ.copy()
    env["DKI_DATAMATE_RUNTIME"] = str(runtime_root)
    env["DKI_USE_DATAMATE_RUNTIME"] = "1"
    env["DKI_REQUIRE_REAL_DATAMATE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=os.fspath(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


# ---- 测试用算子（注册进全局 OPERATORS，名字加前缀避免撞名） ----
class _TestUpperMapper(Mapper):
    def execute(self, sample):
        sample[self.text_key] = sample[self.text_key].upper()
        return sample


class _TestBoomMapper(Mapper):
    """fileId 含 'bad' 时抛异常，测失败隔离。"""

    def execute(self, sample):
        if "bad" in str(sample.get("fileId")):
            raise ValueError("boom")
        return sample


def _ensure_registered():
    if "_TestUpperMapper" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestUpperMapper", module_cls=_TestUpperMapper)
    if "_TestBoomMapper" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestBoomMapper", module_cls=_TestBoomMapper)


@pytest.fixture()
def runner(tmp_path):
    _ensure_registered()
    r = LocalPipelineRunner(load_builtin_ops=False, register_dki_ops=False)
    r.registry.task_dir = tmp_path / "tasks"
    return r


def test_mapper_chain_and_metrics(runner):
    samples = [make_sample(text="abc", file_id="a"), make_sample(text="xyz", file_id="b")]
    res = runner.run([{"_TestUpperMapper": {}}], samples, name="t")
    assert res.report.status == TaskStatus.COMPLETED
    assert [s["text"] for s in res.ok_samples] == ["ABC", "XYZ"]
    m = res.report.op_metrics[0]
    assert m.n_in == 2 and m.n_out == 2 and m.n_failed == 0
    assert m.throughput_per_s() > 0


def test_failure_isolation_partial_success(runner):
    samples = [make_sample(text="x", file_id="good"), make_sample(text="x", file_id="bad1")]
    res = runner.run(
        [{"_TestBoomMapper": {}}, {"_TestUpperMapper": {}}], samples, name="t"
    )
    # 坏样本被隔离但仍在输出流中，且带失败标记；好样本继续被后续算子处理
    assert res.report.status == TaskStatus.PARTIAL_SUCCESS
    assert res.report.n_failed == 1 and res.report.n_output == 1
    ok = res.ok_samples[0]
    assert ok["text"] == "X"
    failed = [s for s in res.samples if s.get(Fields.result) is False][0]
    assert failed["failed_reason"]["op_name"] == "_TestBoomMapper"
    # 第二个算子应跳过失败样本（n_in 含它，但不执行）
    m2 = res.report.op_metrics[1]
    assert m2.n_in == 2 and m2.n_failed == 0


def test_retry_recovers_transient(runner):
    calls = {"n": 0}

    class _Flaky(Mapper):
        def execute(self, sample):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return sample

    if "_TestFlaky" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestFlaky", module_cls=_Flaky)
    res = runner.run([{"_TestFlaky": {}}], [make_sample(text="x", file_id="a")], retry=1)
    assert res.report.status == TaskStatus.COMPLETED
    assert res.report.op_metrics[0].n_retried == 1


def test_runner_flushes_and_closes_owned_operator(runner):
    events = []

    class _Lifecycle(Mapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            events.append(("flags", self.is_first_op, self.is_last_op))

        def execute(self, sample):
            events.append("execute")
            return sample

        def flush(self):
            events.append("flush")
            return True

        def close(self):
            events.append("close")

    if "_TestLifecycle" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestLifecycle", module_cls=_Lifecycle)
    res = runner.run([{"_TestLifecycle": {}}], [make_sample(text="x", file_id="a")])
    assert res.report.status == TaskStatus.COMPLETED
    assert events == [("flags", False, True), "execute", "flush", "close"]


def test_partial_operator_construction_is_unwound(runner):
    events = []

    class _Owned(Mapper):
        def close(self):
            events.append("close")

    class _ConstructorBoom(Mapper):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("constructor boom")

    if "_TestOwned" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestOwned", module_cls=_Owned)
    if "_TestConstructorBoom" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestConstructorBoom", module_cls=_ConstructorBoom)

    result = runner.run(
        [{"_TestOwned": {}}, {"_TestConstructorBoom": {}}],
        [make_sample(text="x", file_id="a")],
    )
    assert result.report.status == TaskStatus.FAILED
    assert events == ["close"]


def test_flush_failure_isolated_and_close_still_runs(runner):
    events = []

    class _FlushBoom(Mapper):
        def execute(self, sample):
            return sample

        def flush(self):
            raise RuntimeError("sidecar flush")

        def close(self):
            events.append("close")

    if "_TestFlushBoom" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestFlushBoom", module_cls=_FlushBoom)
    result = runner.run([{"_TestFlushBoom": {}}], [make_sample(text="x", file_id="a")])
    assert result.report.status == TaskStatus.COMPLETED
    assert result.ok_samples[0]["text"] == "x"
    assert events == ["close"]


def test_execute_batch_preserves_order_and_isolates_item_failure(runner):
    calls = []

    class _Batch(Mapper):
        def execute(self, sample):
            raise AssertionError("per-sample path should not run")

        def execute_batch(self, samples):
            calls.append([sample["fileId"] for sample in samples])
            outputs = []
            for sample in reversed(samples):
                sample["text"] = sample["text"].upper()
                outputs.insert(0, RuntimeError("bad item") if sample["fileId"] == "bad" else sample)
            return outputs

    if "_TestBatch" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestBatch", module_cls=_Batch)
    result = runner.run(
        [{"_TestBatch": {}}],
        [make_sample(text="a", file_id="one"), make_sample(text="b", file_id="bad"), make_sample(text="c", file_id="three")],
    )
    assert calls == [["one", "bad", "three"]]
    assert [sample["fileId"] for sample in result.samples] == ["one", "bad", "three"]
    assert [sample["text"] for sample in result.ok_samples] == ["A", "C"]
    assert result.report.op_metrics[0].n_failed == 1


def test_execute_batch_contract_failure_falls_back_without_partial_mutation(runner):
    class _BrokenBatch(Mapper):
        def execute_batch(self, samples):
            samples[0]["text"] = "CORRUPTED"
            return []

        def execute(self, sample):
            sample["text"] += "!"
            return sample

    if "_TestBrokenBatch" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestBrokenBatch", module_cls=_BrokenBatch)
    result = runner.run(
        [{"_TestBrokenBatch": {}}],
        [make_sample(text="clean", file_id="one")],
    )
    assert result.ok_samples[0]["text"] == "clean!"
    assert result.report.status == TaskStatus.COMPLETED


def test_execute_batch_skips_prior_failed_samples(runner):
    seen = []

    class _BatchAfterFailure(Mapper):
        def execute_batch(self, samples):
            seen.extend(sample["fileId"] for sample in samples)
            return samples

    if "_TestBatchAfterFailure" not in OPERATORS.modules:
        OPERATORS.register_module(module_name="_TestBatchAfterFailure", module_cls=_BatchAfterFailure)
    result = runner.run(
        [{"_TestBoomMapper": {}}, {"_TestBatchAfterFailure": {}}],
        [make_sample(text="a", file_id="good"), make_sample(text="b", file_id="bad")],
    )
    assert seen == ["good"]
    assert [sample["fileId"] for sample in result.samples] == ["good", "bad"]


def test_task_level_error_takes_precedence_over_partial_sample_counts():
    from dki.pipeline.state import TaskReport

    report = TaskReport.new()
    report.n_failed = 1
    report.n_output = 1
    report.error = "pipeline aborted"
    report.mark_finished()
    assert report.status == TaskStatus.FAILED


def test_task_registry_rejects_path_traversal(runner, tmp_path):
    (tmp_path / "secret.json").write_text('{"secret": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid task_id"):
        runner.registry.get("../secret")


def test_task_registry_persistence(runner):
    res = runner.run([{"_TestUpperMapper": {}}], [make_sample(text="a", file_id="a")])
    data = runner.registry.get(res.report.task_id)
    assert data is not None and data["status"] == "COMPLETED"
    assert (runner.registry.task_dir / f"{res.report.task_id}.json").exists()


def test_unknown_operator_fails_task(runner):
    res = runner.run([{"NoSuchOp": {}}], [make_sample(text="a", file_id="a")])
    assert res.report.status == TaskStatus.FAILED
    assert "未注册" in res.report.error


def test_config_normalize():
    assert _normalize(["A", {"B": {"x": 1}}, {"op": "C", "params": {"y": 2}}]) == [
        {"A": {}},
        {"B": {"x": 1}},
        {"C": {"y": 2}},
    ]


def test_datamate_builtin_filter_semantics():
    """Length filtering works with both DataMate and the Windows shim."""
    from dki import dmcompat

    assert dmcompat.load_builtin_ops()
    r = LocalPipelineRunner(register_dki_ops=False)
    samples = [
        make_sample(text="短", file_id="short"),
        make_sample(text="这是一段足够长的文本" * 5, file_id="long"),
    ]
    res = r.run(
        [{"FileWithShortOrLongLengthFilter": {"fileLength": [10, 10000]}}], samples
    )
    ids = [s["fileId"] for s in res.ok_samples]
    assert ids == ["long"]
    assert res.report.op_metrics[0].n_filtered == 1


def test_910c_find_datamate_runtime_reads_env_each_call(tmp_path, monkeypatch):
    from dki import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path / "project")
    first = _make_runtime_root(tmp_path / "runtime-one")
    second = _make_runtime_root(tmp_path / "runtime-two")

    monkeypatch.setenv("DKI_DATAMATE_RUNTIME", str(first))
    assert paths.find_datamate_runtime() == first.resolve()
    monkeypatch.setenv("DKI_DATAMATE_RUNTIME", str(second))
    assert paths.find_datamate_runtime() == second.resolve()


def test_910c_find_datamate_runtime_requires_ops_and_executor(
    tmp_path, monkeypatch
):
    from dki import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path / "project")
    only_ops = tmp_path / "only-ops"
    (only_ops / "ops").mkdir(parents=True)
    monkeypatch.setenv("DKI_DATAMATE_RUNTIME", str(only_ops))
    assert paths.find_datamate_runtime() is None

    only_executor = tmp_path / "only-executor"
    (only_executor / "python-executor").mkdir(parents=True)
    monkeypatch.setenv("DKI_DATAMATE_RUNTIME", str(only_executor))
    assert paths.find_datamate_runtime() is None

    valid = _make_runtime_root(tmp_path / "valid-runtime")
    monkeypatch.setenv("DKI_DATAMATE_RUNTIME", str(valid))
    assert paths.find_datamate_runtime() == valid.resolve()


def test_910c_real_datamate_bootstrap_prepends_executor(tmp_path):
    repo_root = os.fspath(Path(__file__).resolve().parents[1])
    runtime = _make_runtime_root(tmp_path / "runtime")
    _write_fake_datamate(
        runtime,
        """
        import sys

        BOOTSTRAP_SYS_PATH_0 = sys.path[0]

        class _Registry:
            modules = {}

        OPERATORS = _Registry()

        class BaseOp:
            pass

        Filter = LLM = Mapper = Slicer = FileExporter = BaseOp
        """,
    )
    result = _run_dmcompat_subprocess(
        runtime,
        f"""
        import json
        import sys

        sys.path.insert(0, {repo_root!r})
        import dki.dmcompat as dmcompat
        from datamate.core.base_op import BOOTSTRAP_SYS_PATH_0

        print(json.dumps({{
            "available": dmcompat.DATAMATE_AVAILABLE,
            "bootstrap_path": BOOTSTRAP_SYS_PATH_0,
        }}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["available"] is True
    assert payload["bootstrap_path"] == str(
        (runtime / "python-executor").resolve()
    )


def test_910c_require_real_datamate_rejects_shim(tmp_path):
    repo_root = os.fspath(Path(__file__).resolve().parents[1])
    runtime = _make_runtime_root(tmp_path / "runtime")
    _write_fake_datamate(
        runtime,
        """
        raise ImportError("910C sentinel DataMate import failure")
        """,
    )
    result = _run_dmcompat_subprocess(
        runtime,
        f"""
        import sys

        sys.path.insert(0, {repo_root!r})
        import dki.dmcompat
        """,
    )

    assert result.returncode != 0
    assert "DKI_REQUIRE_REAL_DATAMATE=1" in result.stderr
    assert "DKI_DATAMATE_RUNTIME=" in result.stderr
    assert "python-executor" in result.stderr
    assert "910C sentinel DataMate import failure" in result.stderr
