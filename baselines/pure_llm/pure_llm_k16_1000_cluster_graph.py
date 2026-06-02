#!/usr/bin/env python3
"""k16-1000 pure LLM cluster graph (same pipeline as k16-500; uniform passset 1000 under pure_llm/data/)."""

from __future__ import annotations

import sys

from pure_llm_cluster_graph import main

if __name__ == "__main__":
    main(["--k", "16", "--n-events", "1000", *sys.argv[1:]])
