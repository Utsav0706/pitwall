"""Run Prometheus or Grafana locally (no Docker), with this repo's config as code.

    python scripts/monitoring.py prometheus     # http://127.0.0.1:9090
    python scripts/monitoring.py grafana        # http://127.0.0.1:3000, "Pitwall router"

The programs themselves are the official Windows releases, unpacked once (outside the
repo). Point to them with PROMETHEUS_HOME / GRAFANA_HOME if they live elsewhere.
Data goes to .cache/ (git-ignored). Both listen on 127.0.0.1 only. Grafana opens
read-only for anonymous visitors, so no password is ever needed to look at it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_HOME = Path(
    os.environ.get("PROMETHEUS_HOME", "C:/Tools/prometheus-3.15.1.windows-amd64")
)
GRAFANA_HOME = Path(os.environ.get("GRAFANA_HOME", "C:/Tools/grafana-13.3.0"))


def prometheus() -> int:
    exe = PROMETHEUS_HOME / ("prometheus.exe" if sys.platform == "win32" else "prometheus")
    return subprocess.call(
        [
            str(exe),
            f"--config.file={ROOT / 'monitoring/prometheus/prometheus.yml'}",
            f"--storage.tsdb.path={ROOT / '.cache/prometheus'}",
            "--web.listen-address=127.0.0.1:9090",
        ],
        cwd=ROOT,
    )


def grafana() -> int:
    exe = GRAFANA_HOME / "bin" / ("grafana.exe" if sys.platform == "win32" else "grafana")
    env = {
        **os.environ,
        "PROMETHEUS_URL": "http://127.0.0.1:9090",
        "DASHBOARDS_DIR": str(ROOT / "monitoring/grafana/dashboards"),
        "GF_PATHS_PROVISIONING": str(ROOT / "monitoring/grafana/provisioning"),
        "GF_PATHS_DATA": str(ROOT / ".cache/grafana"),
        "GF_PATHS_LOGS": str(ROOT / ".cache/grafana/logs"),
        "GF_SERVER_HTTP_ADDR": "127.0.0.1",
        "GF_SERVER_HTTP_PORT": "3000",
        "GF_AUTH_ANONYMOUS_ENABLED": "true",
        "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
        "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH": str(
            ROOT / "monitoring/grafana/dashboards/pitwall.json"
        ),
        "GF_ANALYTICS_REPORTING_ENABLED": "false",
        "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
        "GF_NEWS_NEWS_FEED_ENABLED": "false",
    }
    return subprocess.call(
        [str(exe), "server", f"--homepath={GRAFANA_HOME}"], cwd=GRAFANA_HOME, env=env
    )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("prometheus", "grafana"):
        sys.exit("usage: python scripts/monitoring.py prometheus|grafana")
    sys.exit(prometheus() if which == "prometheus" else grafana())
