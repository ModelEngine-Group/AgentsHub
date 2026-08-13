"""Import-safe package marker for sealed script entrypoint closure analysis.

Standalone scripts remain directly executable.  The marker only lets the
NL2SQL execution-bundle analyzer bind their static local imports as modules.
"""
