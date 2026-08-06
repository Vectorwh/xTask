#!/usr/bin/env python3
"""Compatibility wrapper for the spaced historical filename.

Prefer: python3 tools/generate_hccl_ranktable.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from xtask.utils.hccl_ranktable import main


if __name__ == "__main__":
    sys.exit(main())
