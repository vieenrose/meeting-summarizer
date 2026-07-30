"""Pack a converted .tflite into a .litertlm bundle for VoxSumDroid.

Usage: export/.venv/bin/python export/pack_litertlm.py <model.tflite> <checkpoint_dir> <out.litertlm>
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai-edge-torch"))
from litert_torch.generative.utilities import litertlm_builder

tflite_path, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
with tempfile.TemporaryDirectory() as workdir:
    litertlm_builder.build_litertlm(
        tflite_model_path=tflite_path,
        workdir=workdir,
        output_path=str(Path(out).parent),
        context_length=32768,
        hf_tokenizer_model_path=str(Path(ckpt) / "tokenizer.json"),
        llm_model_type="qwen3",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
    )
print("packed")
