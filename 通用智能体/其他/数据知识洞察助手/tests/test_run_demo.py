import io
from contextlib import redirect_stdout

import run_demo


def test_print_outputs_is_safe_for_gbk_console():
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="gbk", errors="strict")

    with redirect_stdout(stream):
        run_demo._print_outputs()

    stream.flush()
