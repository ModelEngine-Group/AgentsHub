from __future__ import annotations

from scripts.run_remote_pytest_evidence import _junit_skips


def test_junit_skip_records_node_ids_and_reasons(tmp_path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<testsuite>
  <testcase classname=\"tests.test_sample\" name=\"test_ok\" />
  <testcase classname=\"tests.test_sample\" name=\"test_skipped\">
    <skipped message=\"requires an accelerator\" />
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    assert _junit_skips(junit) == [
        {
            "node_id": "tests.test_sample::test_skipped",
            "reason": "requires an accelerator",
        }
    ]
