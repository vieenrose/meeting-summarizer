"""K2-Horizon (Llama, YaRN RoPE, untied lm_head) checkpoint -> LiteRT tflite (prefill/decode).

Same flags as litert_torch's converters. The rope cache reproduces transformers' YaRN: inv_freq from
`_compute_yarn_parameters`, cos/sin scaled by its attention factor. Needs the tflite schema shim on PYTHONPATH.
    python distill/export_k2_litert.py --checkpoint_path runs/sft/k2-dpo-big/lora/epoch0 --output_path runs/lt/k2 \
        --output_name_prefix k2 --quantize dynamic_int4_block32 --kv_cache_max_len 4096 --prefill_seq_lens 128
"""
import json, os
from absl import app, flags
import torch
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.utilities import converter, model_builder
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

converter.define_conversion_flags('k2', default_mask_as_input=True, default_transpose_kv_cache=True)
FLAGS = flags.FLAGS
converter.create_quantize_suffix = lambda q: {'none': 'f32', 'dynamic_int8': 'q8', 'dynamic_int4_block32': 'q4b32'}.get(q, q)


def config_for(path):
    hf = json.load(open(os.path.join(path, 'config.json')))
    pc = PretrainedConfig.from_dict(hf)
    inv_freq, att = ROPE_INIT_FUNCTIONS['yarn'](pc, 'cpu')

    def rope(input_pos, n_elem, base, **_):
        t = torch.outer(input_pos.float(), inv_freq.float())
        return torch.cos(t) * att, torch.sin(t) * att

    norm = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM, epsilon=hf['rms_norm_eps'])
    block = cfg.TransformerBlockConfig(
        attn_config=cfg.AttentionConfig(num_heads=hf['num_attention_heads'], head_dim=hf['head_dim'],
                                        num_query_groups=hf['num_key_value_heads'], rotary_base=10000, rotary_percentage=1.0),
        ff_config=cfg.FeedForwardConfig(type=cfg.FeedForwardType.GATED, activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
                                        intermediate_size=hf['intermediate_size']),
        pre_attention_norm_config=norm, post_attention_norm_config=norm)
    return cfg.ModelConfig(vocab_size=hf['vocab_size'], num_layers=hf['num_hidden_layers'], max_seq_len=8192,
                           embedding_dim=hf['hidden_size'], block_configs=block, final_norm_config=norm,
                           lm_head_share_weight_with_embedding=False, build_rope=rope)


def build(checkpoint_path, custom_loader=None, mask_cache_size=0):
    return model_builder.build_decoder_only_model(
        checkpoint_path=checkpoint_path, config=config_for(checkpoint_path),
        tensor_names=model_builder.TENSOR_NAMES_WITH_SEPARATE_LM_HEAD, model_class=model_builder.DecoderOnlyModel,
        custom_loader=custom_loader, mask_cache_size=mask_cache_size)


def main(_):
    converter.build_and_convert_to_tflite_from_flags(build)


if __name__ == '__main__':
    app.run(main)
