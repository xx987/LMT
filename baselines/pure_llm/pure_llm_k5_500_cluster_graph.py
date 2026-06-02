#!/usr/bin/env python3
"""Backward-compatible entry: k5-500 pure LLM cluster graph. Delegates to pure_llm_cluster_graph."""

from __future__ import annotations

import sys

from pure_llm_cluster_graph import main

if __name__ == "__main__":
    main(["--k", "5", "--n-events", "500", *sys.argv[1:]])
