import logging
import os
import shutil
from pathlib import Path
from typing import Union
from unsloth import FastLanguageModel
from huggingface_hub import hf_hub_download, list_repo_files

def convert_to_gguf(model_id: str, output_dir: str, quant_method: str, rename_as: str = None):
    """
    Convert a HuggingFace model to GGUF format with optional renaming.
    """
    logging.info(f"🔍 Loading model from {model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(model_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"⚙️ Saving GGUF to {output_dir} with quantization: {quant_method}")
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quant_method)

    gguf_files = list(output_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError("❌ No GGUF file created.")

    if rename_as:
        renamed_path = output_dir / f"{rename_as}.gguf"
        if renamed_path.exists():
            logging.warning(f"⚠️ File {renamed_path} exists. Overwriting.")
            renamed_path.unlink()
        gguf_files[0].rename(renamed_path)
        logging.info(f"✅ Renamed GGUF to: {renamed_path}")

def is_gguf_model(model_id: str) -> bool:
    """
    Check if the model ID likely refers to a GGUF file.
    """
    return model_id.endswith(".gguf") or "GGUF" in model_id.upper()

def check_gguf_exists_in_repo(model_id: str, filename: str) -> bool:
    """
    Check if GGUF file exists in the HuggingFace repository.
    """
    try:
        return filename in list_repo_files(model_id)
    except Exception:
        return False

def download_or_convert_model(
    model_id: str,
    output_path: Union[str, Path],
    quant_method: str = "q4_k_m",
    convert: bool = False,
    rename_as: str = None,
):
    """
    Try to download a GGUF file from HuggingFace.
    If not available, convert from a HuggingFace model using Unsloth.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logging.info(f"✅ GGUF already exists at {output_path}")
        return output_path

    filename = output_path.name

    # Try to download directly if convert is False
    if not convert:
        try:
            logging.info(f"⬇️ Trying to download GGUF: {filename} from {model_id}")
            if check_gguf_exists_in_repo(model_id, filename):
                downloaded = hf_hub_download(repo_id=model_id, filename=filename)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Path(downloaded).replace(output_path)
                logging.info(f"✅ Downloaded GGUF to {output_path}")
                return output_path
            else:
                logging.warning(f"⚠️ GGUF file {filename} not found in repo {model_id}")
                convert = True  # fallback
        except Exception as e:
            logging.warning(f"⚠️ Download failed: {e}")
            convert = True

    # Convert model to GGUF
    if convert:
        try:
            logging.info(f"🛠️ Converting model {model_id} to GGUF")
            model, tokenizer = FastLanguageModel.from_pretrained(model_id)

            temp_dir = output_path.parent / "temp_conversion"
            temp_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained_gguf(temp_dir, tokenizer, quantization_method=quant_method)
            gguf_files = list(temp_dir.glob("**/*.gguf"))

            if not gguf_files:
                raise FileNotFoundError("❌ No GGUF file found after conversion.")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            gguf_files[0].rename(output_path)
            shutil.rmtree(temp_dir)

            logging.info(f"✅ Converted and saved GGUF to {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"❌ Conversion failed: {e}")
            raise