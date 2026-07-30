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
    # The llm_model_type enum alone does NOT make the runtime apply a chat template —
    # the prompt_templates fields do (verified: without them the engine doc-completes
    # the transcript in a loop). Qwen3 ChatML, non-thinking, matching training.
    litertlm_builder.build_litertlm(
        tflite_model_path=tflite_path,
        workdir=workdir,
        output_path=str(Path(out).parent),
        context_length=32768,
        hf_tokenizer_model_path=str(Path(ckpt) / "tokenizer.json"),
        llm_model_type="qwen3",
        user_prompt_prefix="<|im_start|>user\n",
        user_prompt_suffix="<|im_end|>\n",
        model_prompt_prefix="<|im_start|>assistant\n",
        model_prompt_suffix="<|im_end|>\n",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
    )
print("packed")
