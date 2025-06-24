from typing import Any, Dict, List
import time
import logging
import subprocess
import requests
from openai import OpenAI
from ..base_model import BaseModel
from .script_local import ScriptLocal
from ..utils.utils import download_or_convert_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

class LLamaCPP(BaseModel):
    """
    Manages a local llama.cpp model server and handles OpenAI-compatible chat completions.
    """

    def __init__(self, script_local: ScriptLocal) -> None:
        self.script_local = script_local
        self.model_name_or_path = script_local.model_name_or_path
        self.process = None

    def download_model(self, quant_method: str = "q4_k_m", convert: bool = False, rename_as: str = None):
        """
        Download or convert a model to GGUF format.
        """
        return download_or_convert_model(
            model_id=self.script_local.model_id,
            output_path=self.script_local.gguf_path,
            quant_method=quant_method,
            convert=convert,
            rename_as=rename_as,
        )

    def load_model(self):
        """
        Start llama.cpp server and initialize OpenAI-compatible client.
        """
        self.script_local.gguf_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.script_local.gguf_path.exists():
            logging.warning(f"⚠️ Model not found at {self.script_local.gguf_path}. Attempting download...")
            try:
                self.download_model(
                    quant_method=self.script_local.dtype,
                    convert=not str(self.script_local.model_name_or_path).lower().endswith(".gguf"),
                    rename_as=self.script_local.gguf_path.stem
                )
            except Exception as e:
                logging.error(f"❌ Model preparation failed: {e}")
                raise FileNotFoundError("Model could not be downloaded or converted.")

        if not self.script_local.server_bin_path.exists():
            raise FileNotFoundError(f"❌ llama.cpp binary not found at: {self.script_local.server_bin_path}")

        logging.info(f"🚀 Starting llama.cpp server: {self.script_local.gguf_path.name}")
        self.process = subprocess.Popen(
            [
                str(self.script_local.server_bin_path),
                "--model", str(self.script_local.gguf_path),
                "--port", str(self.script_local.port),
                "--host", "0.0.0.0",
                "--threads", str(self.script_local.n_thread),
                "--n_ctx", "8192",
                "--n_batch", "512",
                "--mlock"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        logging.info("📄 Waiting for model to be ready...")
        self.client = OpenAI(
            base_url=f"http://localhost:{self.script_local.port}/v1",
            api_key="sk-no-auth"
        )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout_sec: int = 1200):
        """
        Wait until the server responds to a test chat completion request.
        """
        waited = 0
        start_time = time.time()
        test_messages = [
            {"role": "system", "content": "ping"},
            {"role": "user", "content": "ping"}
        ]

        while waited < timeout_sec:
            if self.process.poll() is not None:
                raise RuntimeError("❌ llama.cpp server crashed before becoming ready.")

            try:
                response = requests.post(
                    f"http://localhost:{self.script_local.port}/v1/chat/completions",
                    json={
                        "model": self.script_local.gguf_path.name,
                        "messages": test_messages,
                        "max_tokens": 1
                    },
                    timeout=10,
                )

                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    logging.info(f"✅ Model ready (loaded in {elapsed:.1f} seconds)")
                    return
                elif response.status_code == 503:
                    logging.debug("⏳ Model still loading...")
                else:
                    logging.warning(f"⚠️ Unexpected status {response.status_code}: {response.text}")
            except Exception as e:
                logging.debug(f"⏳ Waiting for server... ({e})")

            time.sleep(5)
            waited += 5

        raise TimeoutError("❌ llama.cpp server did not respond within timeout.")

    def generate(self, inputs: List[Dict[str, str]], max_tokens: int = 256, stream: bool = True) -> str:
        """
        Send chat completion request to the local llama.cpp server.
        """
        try:
            response_text = ""
            completion = self.client.chat.completions.create(
                model=self.script_local.gguf_path.name,
                messages=inputs,
                max_tokens=max_tokens,
                stream=stream,
            )

            if stream:
                for chunk in completion:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end='', flush=True)
                        response_text += delta
                print()
            else:
                response_text = completion.choices[0].message.content
                print(response_text)

            return response_text

        except Exception as e:
            logging.error(f"❌ Inference failed: {e}")
            raise

    def stop(self):
        """
        Stop the llama.cpp server process.
        """
        if self.process:
            self.process.terminate()
            self.process.wait()
            logging.info("🛑 llama.cpp server stopped.")
