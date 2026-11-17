"""SDXL DAAM tracing via a custom AttnProcessor (diffusers >= 0.19).

The vendored DAAM fork hooks the legacy ``CrossAttention._attention`` API
(diffusers 0.9.0, used for SD 1.4/1.5). SDXL runs on newer diffusers where
attention is delegated to an ``AttnProcessor`` (default ``AttnProcessor2_0``
uses fused ``scaled_dot_product_attention`` and exposes no weights). This
module installs a processor that computes cross-attention explicitly so the
weights can be recorded, then aggregates them into the same ``GlobalHeatMap``
the rest of DAAM consumes. ``heatmap.GlobalHeatMap`` / ``WordHeatMap`` and
``box_extract`` are reused unchanged.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .heatmap import GlobalHeatMap
from .utils import auto_autocast

__all__ = ['trace_xl', 'DiffusionXlHeatMapHooker']


def _closest_factor(x):
    for i in range(int(math.sqrt(x)), 0, -1):
        if x % i == 0:
            return i, x // i


class _DaamAttnProcessor:
    """Attention processor that records cross-attention weights for DAAM.

    Mirrors diffusers' AttnProcessor but keeps the explicit softmax attention
    matrix (instead of fused SDPA) so it can be captured. Only cross-attention
    (``encoder_hidden_states`` present) is recorded.
    """

    def __init__(self, hooker):
        self.hooker = hooker

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        b, seq, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        context = encoder_hidden_states if is_cross else hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(context)
        v = attn.to_v(context)
        head_dim = k.shape[-1] // attn.heads
        q = q.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, attn.heads, head_dim).transpose(1, 2)

        aw = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            aw = aw + attention_mask
        aw = aw.softmax(dim=-1)

        if is_cross:
            self.hooker._record(aw)

        hs = torch.matmul(aw, v)
        hs = hs.transpose(1, 2).reshape(b, -1, attn.heads * head_dim)
        hs = attn.to_out[0](hs)
        hs = attn.to_out[1](hs)
        if input_ndim == 4:
            hs = hs.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hs = hs + residual
        hs = hs / attn.rescale_output_factor
        return hs


class DiffusionXlHeatMapHooker:
    def __init__(self, pipeline):
        self.pipe = pipeline
        self.last_prompt = ''
        self._attn = []          # per cross-attn call: (hw, tokens) cond map, head-averaged
        self._saved = {}
        for name, mod in pipeline.unet.named_modules():
            if hasattr(mod, 'to_q') and hasattr(mod, 'set_processor'):
                self._saved[name] = mod.processor
                mod.set_processor(_DaamAttnProcessor(self))

    def _record(self, aw):
        # aw: (batch, heads, hw, tokens). With CFG batch=[uncond, cond]; keep cond.
        cond = aw[1] if aw.shape[0] == 2 else aw[0]
        self._attn.append(cond.mean(0).detach().float().cpu())

    def restore(self):
        for name, mod in self.pipe.unet.named_modules():
            if name in self._saved:
                mod.set_processor(self._saved[name])

    def compute_global_heat_map(self, prompt=None, **kwargs):
        prompt = prompt or self.last_prompt
        if not self._attn:
            raise RuntimeError('No cross-attention captured; generate inside `with trace_xl(pipe)`.')
        # SDXL's largest cross-attention sits at latent/2, so latent side = 2*sqrt(max_hw).
        max_hw = max(m.shape[0] for m in self._attn)
        target = 2 * int(math.sqrt(max_hw))
        with auto_autocast(dtype=torch.float32):
            merges = []
            for m in self._attn:
                hw, tokens = m.shape
                w, h = _closest_factor(hw)
                mm = m.permute(1, 0).reshape(tokens, h, w).unsqueeze(1)
                merges.append(F.interpolate(mm, size=(target, target), mode='bicubic').clamp_(min=0))
            maps = torch.stack(merges).mean(0)[:, 0]      # (tokens, target, target)
            maps = maps[:len(self.pipe.tokenizer.tokenize(prompt)) + 2]
        return GlobalHeatMap(self.pipe.tokenizer, prompt, maps)


@contextmanager
def trace_xl(pipeline):
    hooker = DiffusionXlHeatMapHooker(pipeline)
    orig_encode = pipeline.encode_prompt

    def _encode(*a, **kw):
        p = kw.get('prompt', a[0] if a else None)
        hooker.last_prompt = p if isinstance(p, str) else (p[0] if p else '')
        hooker._attn.clear()
        return orig_encode(*a, **kw)

    pipeline.encode_prompt = _encode
    try:
        yield hooker
    finally:
        hooker.restore()
        pipeline.encode_prompt = orig_encode
