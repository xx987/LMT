#!/usr/bin/env python3
"""k16-500 pure LLM cluster graph (same pipeline as k5; cascadefriendly data under pure_llm/data/)."""

from __future__ import annotations

import sys

from pure_llm_cluster_graph import main

if __name__ == "__main__":
    main(["--k", "16", "--n-events", "500", *sys.argv[1:]])
