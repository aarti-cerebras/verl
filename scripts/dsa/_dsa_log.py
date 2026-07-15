# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared logging setup for the DSA trajectory-generation scripts.

Every DSA script calls ``setup_logging(name, log_dir)`` at startup. It writes to BOTH stdout and a
timestamped log file under ``log_dir``, and records — as the first lines of the log — the **exact command
that was run** (``sys.executable`` + ``sys.argv``), the cwd/host, the git commit, and the relevant env vars,
so any log file is self-describing and the run is reproducible. Import-path safe: scripts add their own dir
to ``sys.path`` before importing this.
"""

import logging
import os
import socket
import subprocess
import sys
import time


def setup_logging(name: str, log_dir: str) -> tuple[logging.Logger, str]:
    """Create a logger writing to ``<log_dir>/<name>_<UTC-timestamp>.log`` + stdout; log the invocation."""
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = os.path.join(log_dir, f"{name}_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # idempotent if called twice in one process
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("========== %s ==========", name)
    logger.info("cmd: %s", " ".join([sys.executable] + sys.argv))
    logger.info("cwd: %s", os.getcwd())
    logger.info("host: %s  pid: %d", socket.gethostname(), os.getpid())
    for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HOME", "HF_TOKEN_PATH", "VLLM_WORKER_MULTIPROC_METHOD"):
        if os.environ.get(k):
            logger.info("env %s=%s", k, os.environ[k])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        logger.info("git: %s", commit.decode().strip())
    except Exception:
        pass
    logger.info("log_file: %s", log_path)
    return logger, log_path
