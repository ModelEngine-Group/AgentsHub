"""
pipelines
=========
三个任务的 Pipeline 入口。
"""

from .task1_pipeline import run as task1_run
from .task1_pipeline import run as task2_run  # TODO: task2
from .task1_pipeline import run as task3_run  # TODO: task3

__all__ = ["task1_run", "task2_run", "task3_run"]
