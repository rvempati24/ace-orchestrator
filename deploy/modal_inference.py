"""Scale-to-zero Holo inference on Modal. Import/deploy explicitly; never run from tests."""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "ace-orchestrator-inference"
MODEL_NAME = os.environ.get("ACE_MODEL_ID", "Hcompany/Holo-3.1-35B-A3B")
MODEL_REVISION = os.environ.get("ACE_MODEL_REVISION", "2bdb92851a8cd9d72cdd891fdf38cfcc7fefae2c")
PORT = 8000
MINUTE = 60

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("ace-orchestrator-model-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ace-orchestrator-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub[hf_transfer]>=0.36,<2")
    .env(
        {
            "HF_HOME": "/models",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)


@app.server(
    image=image,
    gpu="H200:1",
    cpu=8,
    memory=65_536,
    port=PORT,
    startup_timeout=20 * MINUTE,
    scaledown_window=10 * MINUTE,
    target_concurrency=4,
    routing_region="us-east",
    volumes={"/models": model_cache, "/root/.cache/vllm": vllm_cache},
    unauthenticated=False,
)
class HoloServer:
    @modal.enter()
    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL_NAME,
                "--revision",
                MODEL_REVISION,
                "--served-model-name",
                MODEL_NAME,
                "--host",
                "0.0.0.0",
                "--port",
                str(PORT),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "32768",
                "--limit-mm-per-prompt",
                '{"image": 1, "video": 0, "audio": 0}',
                "--enforce-eager",
            ]
        )

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


@app.local_entrypoint()
def endpoint() -> None:
    print(HoloServer.get_url())
