#!/usr/bin/env python3
"""Container/readiness probe used by Docker HEALTHCHECK."""

import json
import os
import sys
import urllib.request

port = os.environ.get("PORT", "8000")
url = f"http://127.0.0.1:{port}/healthz/ready"
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        payload = json.load(resp)
    if payload.get("status") == "ready":
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
