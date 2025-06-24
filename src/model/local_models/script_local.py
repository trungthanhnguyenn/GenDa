from dataclasses import dataclass
from pathlib import Path

@dataclass
class ScriptLocal:
    model_name_or_path: str
    client_url: str
    server_bin_path: Path
    gguf_path: Path
    num_device: int
    dtype: str
    port: int
    model_id: str
    n_thread: int
    # support:str = None
        
