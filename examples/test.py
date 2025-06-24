import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from model.local_models.local_model import LLamaCPP
from model.local_models.script_local import ScriptLocal

def main():
    # Fill these paths and values according to your environment
    script_local = ScriptLocal(
        model_name_or_path="unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
        client_url="http://localhost:8000/v1",
        server_bin_path=Path("llama.cpp/build/bin/llama-server"),
        gguf_path=Path("./models/Llama-4-Maverick-17B-128E-Instruct-UD-TQ1_0.gguf"),
        num_device=1,
        dtype="q4_k_m",
        port=8000,
        model_id="unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
        n_thread = 12
    )

    model = LLamaCPP(script_local)
    try:
        model.load_model()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        print("Generating response...")
        response = model.generate(messages, max_tokens=64, stream=False)
        print("Model response:", response)
    finally:
        # model.stop()
        pass
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()