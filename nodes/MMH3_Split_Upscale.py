"""MMH3 Split Upscale - 完整最终版 (含 seam_denoise 缝邻域降噪上限)
时间: 固定长度重叠窗口 + B 重采样 overlap + latent 线性交叉淡化
校正: 空间/时间两级颜色匹配 + 首块源参考 + 每 chunk 全局钉源
抗分叉: seam_denoise 缝邻域降噪上限 (高降噪+快运动防切割残影)
修复: 每瓦片独立 Guider + 相位0 identity anchors + previous-output motion/warm-up keyframes; probe 门控 polish
定位: 每个 temporal chunk 使用绝对 H3 时间坐标，并按全局 token phase 旋转 1,4,4,4,4 时间步幅
续接: temporal overlap 在每个后续 chunk 中正常重采样，并在最终 latent 中 A→B 交叉淡化
简化: overlap/fade 百分比参数。
"""

import math
import gc
import contextvars
import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.nested_tensor
import comfy.ldm.minimax.model as h3model
import latent_preview
from comfy_api.latest import io

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0


# ---------------------------------------------------------------------------
# Absolute H3 timeline positions
# ---------------------------------------------------------------------------
# ComfyUI's H3 PackedLayout normally starts every independently sampled window
# at the same local target-time origin.  Split Upscale knows the absolute pixel
# frame where each temporal window begins, so publish it through a ContextVar
# around the complete sampler invocation.  PackedLayout is built inside that
# invocation (including ComfyUI conditioning preparation / H3-Optimizations),
# therefore all consumers see one consistent position grid without changing
# sequence length or sparse-layout topology.
_MMH3_WINDOW_START_FRAMES = contextvars.ContextVar(
    "mmh3_window_start_frames", default=0.0)
# Global H3 token index of the current independently sampled window.  Absolute
# frame offset alone is insufficient when a split begins off phase 0, because
# stock H3 restarts its 1,4,4,4,4 token-span cadence at local token 0.
_MMH3_WINDOW_START_TOKEN = contextvars.ContextVar(
    "mmh3_window_start_token", default=0)
_MMH3_ABSOLUTE_TIMELINE_PATCH_OK = False


# ---------------------------------------------------------------------------
# MiniMax H3 Extend-style context support
# ---------------------------------------------------------------------------
def _context_k_distance(k):
    if k >= 0:
        return 0.0
    m = -k
    return sum(h3model.FRAME_RESCALE * h3model.FRAME_PER_TOKEN[(1 - i) % 5]
               for i in range(1, m + 1))


def _packed_layout_supports_context():
    """Feature-test rather than version-test ComfyUI's PackedLayout."""
    try:
        z = torch.empty((1, 24, 1, 2, 2))
        h3model.PackedLayout(
            1, 1, 2, 2, 1,
            keyframes=[{"kind": "context", "num_frames": 1, "latent": z}],
            refs=None,
        )
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _install_h3_context_layout_patch():
    """Add kat3ri-style fixed context rows while preserving current Comfy keyframes.

    Current Comfy already supports arbitrary resolved_frame_index keyframes and
    correctly appends refs in MiniMaxH3.extra_conds.  Only PackedLayout needs
    extension for kind=context/context_audio, so we deliberately do not replace
    MiniMaxH3.extra_conds or regress current ordinary-keyframe behavior.
    """
    if getattr(h3model.PackedLayout, "_mmh3_context_support", False):
        return
    if _packed_layout_supports_context():
        h3model.PackedLayout._mmh3_context_support = True
        return

    def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=None, refs=None):
        frame, w_grid = h3model._frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        row = text_len

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        cursor = float(text_len)
        for blk in refs or ():
            cursor += h3model._ref_t_span(blk)

        # Context video blocks are listed newest-to-oldest if more than one is
        # supplied. Our node emits one contiguous block, but the cursor keeps
        # the patch well-defined for multiple blocks too.
        context_k_cursor = 0
        audio_context_cursor = cursor

        if keyframes:
            for kf in keyframes:
                kind = kf.get("kind")
                if kind == "context":
                    video_latent = kf.get("latent")
                    if video_latent is None:
                        continue
                    n_frames = int(kf.get("num_frames", video_latent.shape[2]))
                    n_frames = min(n_frames, video_latent.shape[2])
                    ks = range(context_k_cursor - n_frames + 1, context_k_cursor + 1)
                    t_grid = torch.tensor(
                        [cursor - _context_k_distance(k) for k in ks],
                        dtype=torch.float64,
                    )
                    context_k_cursor -= n_frames
                    gg = torch.empty(n_frames, frame_rows, 3, dtype=torch.float64)
                    gg[:, :, 0] = t_grid[:, None]
                    gg[:, :, 1:] = frame[None]
                    n = n_frames * frame_rows
                    segments.append(("cond", n))
                    pos.append(gg.reshape(-1, 3))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    continue

                if kind == "context_audio":
                    audio_latent = kf.get("audio_latent")
                    if audio_latent is None:
                        continue
                    rt = int(kf.get("num_frames", audio_latent.shape[-1]))
                    rt = min(rt, audio_latent.shape[-1])
                    segments.append(("cond_audio", rt * 2))
                    pos.append(h3model._audio_grid(
                        audio_context_cursor - rt, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    audio_context_cursor -= rt
                    row += rt * 2
                    continue

                # Preserve current Comfy behavior for every ordinary keyframe,
                # including negative and mid-segment resolved_frame_index values.
                cond_t = cursor + h3model.FRAME_RESCALE * kf["resolved_frame_index"]
                video_latent = kf.get("latent")
                if video_latent is not None:
                    vt = video_latent.shape[2]
                    n = vt * frame_rows
                    segments.append(("cond", n))
                    pos.append(h3model._video_grid(vt, frame, cond_t))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                audio_latent = kf.get("audio_latent")
                if audio_latent is not None:
                    rt = audio_latent.shape[-1]
                    segments.append(("cond_audio", rt * 2))
                    pos.append(h3model._audio_grid(cond_t, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2

        if refs:
            ref_cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = h3model._frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    gg = torch.empty(n, 3, dtype=torch.float64)
                    gg[:, 0] = ref_cursor
                    gg[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(gg)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    ref_cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(h3model._audio_grid(ref_cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    ref_cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = h3model._frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(h3model._audio_grid(
                            ref_cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(h3model._video_grid(vt, r_frame, ref_cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    ref_cursor += max(float(rt), sum(h3model._video_t_spans(vt)))

        segments.append(("audio", audio_t * 2))
        pos.append(h3model._audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(h3model._video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs

    h3model.PackedLayout.__init__ = _patched_init
    h3model.PackedLayout._mmh3_context_support = True
    print("[H3] installed Extend-style context PackedLayout support")


_install_h3_context_layout_patch()


def _install_h3_absolute_timeline_patch():
    """Put independently sampled H3 windows on the true global temporal grid.

    Two corrections are required:

    1. Absolute origin: move timeline-bearing rows by FRAME_RESCALE * the
       window's decoded/pixel-frame start.
    2. H3 cadence phase: stock ``_video_grid`` always starts its temporal spans
       at phase 0 of ``FRAME_PER_TOKEN = (1,4,4,4,4)``.  A split window may
       begin at global token phase 1..4.  For the *target video rows* we replace
       that local phase-0 cumsum with the cumsum beginning at
       ``global_start_token % 5``.

    Only position values change.  Row count, segment boundaries and packed
    topology are untouched, preserving compatibility with sparse/cube-order
    wrappers such as H3-Optimizations.
    """
    global _MMH3_ABSOLUTE_TIMELINE_PATCH_OK
    cls = getattr(h3model, "PackedLayout", None)
    if cls is None:
        print("[H3] WARNING: PackedLayout not found; absolute timeline unavailable")
        return False
    if getattr(cls, "_mmh3_absolute_timeline", False):
        _MMH3_ABSOLUTE_TIMELINE_PATCH_OK = True
        return True

    base_init = cls.__init__

    def _absolute_init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        start_frames = float(_MMH3_WINDOW_START_FRAMES.get())
        start_token = int(_MMH3_WINDOW_START_TOKEN.get())
        phase = start_token % len(h3model.FRAME_PER_TOKEN)
        if abs(start_frames) <= 1e-12 and phase == 0:
            return

        pos = getattr(self, "position_ids", None)
        segments = getattr(self, "segments", None)
        signature = getattr(self, "signature", None)
        if pos is None or segments is None:
            raise RuntimeError(
                "H3 absolute timeline patch requires PackedLayout.position_ids and segments")

        # Move timeline-local conditioning and both target streams to the
        # absolute origin. Reference blocks intentionally stay stock because
        # they are presentation/source material rather than target-time rows.
        dt = float(h3model.FRAME_RESCALE) * start_frames
        if abs(dt) > 1e-12:
            timeline_kinds = {"cond", "cond_audio", "audio", "video"}
            for a, b, kind in segments:
                if kind in timeline_kinds and int(b) > int(a):
                    pos[int(a):int(b), 0].add_(dt)

        # Correct the target video's *within-window* cadence when this window
        # begins off phase 0.  Stock _video_grid produced starts using:
        #   0, span[0], span[0]+span[1], ...
        # We add the difference to the starts produced from the global phase.
        # Spatial rows for each temporal token are contiguous, so each temporal
        # correction is repeated frame_rows times.
        if phase != 0:
            if signature is None or len(signature) < 4:
                raise RuntimeError(
                    "H3 phase-aware absolute timeline patch requires PackedLayout.signature")
            latent_t = int(signature[1])
            video_seg = next(((int(a), int(b)) for a, b, kind in segments
                              if kind == "video"), None)
            if video_seg is None:
                raise RuntimeError("H3 absolute timeline patch could not find target video segment")
            va, vb = video_seg
            n_video_rows = vb - va
            if latent_t <= 0 or n_video_rows % latent_t != 0:
                raise RuntimeError(
                    f"H3 target-video row geometry mismatch: rows={n_video_rows}, t={latent_t}")
            frame_rows = n_video_rows // latent_t

            fpt = tuple(int(v) for v in h3model.FRAME_PER_TOKEN)
            scale = float(h3model.FRAME_RESCALE)
            local_starts = [0.0]
            global_starts = [0.0]
            for k in range(1, latent_t):
                local_starts.append(local_starts[-1] + scale * fpt[(k - 1) % len(fpt)])
                global_starts.append(global_starts[-1] + scale * fpt[(phase + k - 1) % len(fpt)])
            corr = torch.tensor(
                [g - l for g, l in zip(global_starts, local_starts)],
                dtype=pos.dtype, device=pos.device,
            ).repeat_interleave(frame_rows)
            pos[va:vb, 0].add_(corr)

        self._mmh3_window_start_frames = start_frames
        self._mmh3_window_start_token = start_token
        self._mmh3_window_start_phase = phase
        self._mmh3_window_start_rope = dt

    _absolute_init.__name__ = getattr(base_init, "__name__", "__init__")
    _absolute_init.__qualname__ = getattr(base_init, "__qualname__", _absolute_init.__qualname__)
    cls.__init__ = _absolute_init
    cls._mmh3_absolute_timeline = True
    cls._mmh3_absolute_timeline_base_init = base_init
    _MMH3_ABSOLUTE_TIMELINE_PATCH_OK = True
    print("[H3] installed absolute + phase-aware temporal-window position patch for PackedLayout")
    return True


_install_h3_absolute_timeline_patch()


# ---------------------------------------------------------------------------
# Identity-anchor denoising gate
# ---------------------------------------------------------------------------
# Anchor strength now uses ComfyUI's native single global
# ``minimax_visual_cond_noise_aug`` conditioning value.  Only the gate needs a
# custom marker because stock H3 has no per-keyframe lifetime control.
_MMH3_IDENTITY_GATE_KEY = "_mmh3_identity_gate_strength"
_MMH3_IDENTITY_GATE_PATCH_OK = False


def _install_h3_identity_gate_patch():
    """Drop only Split-Upscale identity keyframes after their H3 progress gate.

    This wrapper is deliberately installed on ``MiniMaxH3Model.forward`` rather
    than ``_forward``.  H3-Optimizations wraps ``_forward`` (cube ordering,
    sparse-attention runtime publication, etc.) and computes its packed layout
    *before* calling the wrapped function.  Filtering keyframes inside
    ``_forward`` therefore changes the packed sequence after those wrappers
    have already committed to the old sequence length.

    Filtering here makes every downstream wrapper see the same active payload.
    A marked identity keyframe with gate ``s`` is active while ``t_v < s``.
    Since H3 uses ``t_v = 1 - sigma_v``, it is removed once
    ``sigma_v <= 1 - s``. Other keyframes, refs, target-video rows,
    and audio conditions remain untouched.
    """
    global _MMH3_IDENTITY_GATE_PATCH_OK
    cls = getattr(h3model, "MiniMaxH3Model", None)
    if cls is None:
        print("[H3] WARNING: MiniMaxH3Model not found; identity gating unavailable")
        return False
    if getattr(cls, "_mmh3_identity_gate", False):
        _MMH3_IDENTITY_GATE_PATCH_OK = True
        return True

    base_forward = cls.forward

    def _gated_model_forward(self, x, timestep, context, transformer_options={},
                             minimax_payload=None, denoise_mask=None,
                             audio_denoise_mask=None, **kwargs):
        payload = minimax_payload
        if isinstance(payload, dict):
            keyframes = payload.get("keyframes") or ()
            if any(_MMH3_IDENTITY_GATE_KEY in kf for kf in keyframes):
                sigma_v = float(
                    (timestep.flatten()[0] / 1000.0)
                    .detach().float().clamp(min=1e-6)
                )
                t_v = 1.0 - sigma_v
                active = []
                changed = False
                for kf in keyframes:
                    gate = kf.get(_MMH3_IDENTITY_GATE_KEY)
                    if gate is None:
                        active.append(kf)
                        continue
                    gate = max(0.0, min(1.0, float(gate)))
                    # Pin-then-flip semantics: active while t_v < strength.
                    if t_v < gate:
                        active.append(kf)
                    else:
                        changed = True

                if changed:
                    p = dict(payload)
                    refs = list(payload.get("refs") or ())
                    p["keyframes"] = active
                    # Keep the packed conditioning rows in exactly the same order
                    # as PackedLayout: active keyframes first, refs second.
                    p["cond_video_latents"] = (
                        [kf["latent"] for kf in active if kf.get("latent") is not None]
                        + [ref["latent"] for ref in refs if ref.get("latent") is not None]
                    )
                    p["cond_audio_latents"] = (
                        [kf["audio_latent"] for kf in active if kf.get("audio_latent") is not None]
                        + [ref["audio_latent"] for ref in refs if ref.get("audio_latent") is not None]
                    )
                    # extra_conds normally prebuilds this for the full keyframe set.
                    # Removing it here forces H3/H3-Optimizations to build the layout
                    # from the same filtered keyframe set before cube ordering/sparse
                    # runtime metadata are established.
                    p.pop("layout", None)
                    payload = p

        return base_forward(
            self, x, timestep, context, transformer_options,
            minimax_payload=payload, denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask, **kwargs)

    _gated_model_forward.__name__ = getattr(base_forward, "__name__", "forward")
    _gated_model_forward.__qualname__ = getattr(base_forward, "__qualname__", _gated_model_forward.__qualname__)
    cls.forward = _gated_model_forward
    cls._mmh3_identity_gate = True
    cls._mmh3_identity_gate_base = base_forward
    _MMH3_IDENTITY_GATE_PATCH_OK = True
    print("[H3] installed per-identity-keyframe denoising-time gate (pre-wrapper payload filter)")
    return True


_install_h3_identity_gate_patch()

H3_TEMPORAL_PARAM = io.Custom("H3_TEMPORAL_PARAM")
H3_SPATIAL_PARAM = io.Custom("H3_SPATIAL_PARAM")

VAE_DOWNSAMPLE = 16
ALIGN = 2
MAX_IDENTITY_ANCHORS = 16
POLISH_HALO = 16
DC_MATCH_CLAMP = 0.05
SEAM_CORR_GATE, SEAM_DC_GATE = 0.85, 0.6
COLOR_CLAMP = 0.05

# ---------------------------------------------------------------------------
# 帧/token 映射
# ---------------------------------------------------------------------------
def frames_for_tokens(n): return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))

def tokens_for_frames(f):
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]; n += 1
    return n

def audio_range(f0, f1): return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)

def clip_tokens(n): return (n - 5) // 17 * 5 + 2 if n >= 5 else 1

def snap_clip_frames(v): return 5 + 17 * max(1, round((v - 5) / 17)) if v >= 5 else max(1, int(v))

def snap_overlap_frames(v): return 0 if v <= 0 else 5 + 17 * max(0, round((v - 5) / 17))

def token_start_at_or_before(f):
    k = 0
    while frames_for_tokens(k + 1) <= f:
        k += 1
    return k

def steps_for_frames(n):
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]; k += 1
    return k if covered == n else None

def compute_h3_segments_adaptive(tv, chunk_frames, overlap_frames):
    """Build fixed-size overlapping H3 temporal windows.

    The overlap is part of both neighboring sampling windows.  For 141/22:
      chunk A: 42 sampled tokens
      chunk B: 7 regenerated overlap + 35 fresh = 42 sampled tokens
    Final assembly crossfades A/B across those 7 duplicate latent tokens.
    The overlapped window start stays on phase 0 of H3's 5-token cadence.
    """
    tc = clip_tokens(chunk_frames)
    to = clip_tokens(overlap_frames) if overlap_frames > 0 else 0
    if to >= tc:
        to = max(0, tc - 1)
    hop = max(1, tc - to) if to > 0 else tc

    bounds = []
    if tv <= 0:
        return bounds, 0

    # First clip has no previous overlap and uses the full nominal window.
    fresh0 = 0
    fresh1 = min(tv, tc)
    bounds.append((0, fresh0, fresh1, 0))  # sample_k0, fresh_k0, fresh_k1, warmup_t

    fresh0 = fresh1
    while fresh0 < tv:
        warmup_t = min(to, fresh0)
        sample_k0 = fresh0 - warmup_t
        # On the standard 17k+5 grid (e.g. 141/22), this must be phase 0.
        if warmup_t > 0 and sample_k0 % 5 != 0:
            raise RuntimeError(
                f"H3 temporal-overlap phase mismatch: sample starts at token {sample_k0} "
                f"(phase {sample_k0 % 5}); choose chunk/overlap values that keep "
                "the overlap window on the 5-token H3 cadence."
            )
        fresh1 = min(tv, fresh0 + hop)
        bounds.append((sample_k0, fresh0, fresh1, warmup_t))
        fresh0 = fresh1

    return bounds, frames_for_tokens(tv)

def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)

def px_to_lat(px):
    return max(ALIGN, (round(px / VAE_DOWNSAMPLE) // ALIGN) * ALIGN)

def snap_align(v):
    return max(0, int(round(v / ALIGN)) * ALIGN)

# ---------------------------------------------------------------------------
# keyframe 链路
# ---------------------------------------------------------------------------
def trim_keyframe(kf, f0, f1):
    if kf.get("kind") in ("context", "context_audio"):
        return dict(kf)
    idx = kf["resolved_frame_index"]
    latent, audio_latent = kf.get("latent"), kf.get("audio_latent")
    if latent is None and audio_latent is None:
        return None if (idx < f0 or idx >= f1) else {"resolved_frame_index": idx - f0}
    out = {}
    if latent is not None:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None: t_start = k
                t_end = k + 1
            pos += span
        if t_start is None: return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if audio_latent is not None:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    return out if ("latent" in out or "audio_latent" in out) else None

def reanchor_conditioning(cond, f0, f1, spatial):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [kf for kf in (trim_keyframe(kf, f0, f1) for kf in kfs) if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = torch.nn.functional.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear",
                                align_corners=False).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out

def prepend_keyframes(cond, kfs):
    if not kfs:
        return cond
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd["minimax_keyframes"] = kfs + (nd.get("minimax_keyframes") or [])
        out.append([tensor, nd])
    return out


def set_anchor_strength(cond, strength):
    """Use stock ComfyUI H3's single visual-condition augmentation value.

    This intentionally applies one value to all visual keyframes/references in
    the chunk, matching the pre per-anchor-patch behavior and avoiding runtime
    source rewriting of MiniMaxH3Model._forward.
    """
    value = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd["minimax_visual_cond_noise_aug"] = value
        out.append([tensor, nd])
    return out

def _context_span(n_tokens):
    """H3 cursor-axis duration of trailing latent tokens ending at target t=0."""
    return sum(FRAME_RESCALE * FRAME_PER_TOKEN[(1 - j) % 5]
               for j in range(n_tokens))


def _motion_context_tokens(n_frames):
    n = next((g for g in (56, 39, 22, 5) if g <= n_frames), 0)
    return steps_for_frames(n) if n else 0


def motion_keyframes(prev_video, prev_tokens, f0, n_frames):
    """Restore the original repo's previous-output motion keyframes.

    The selected tail is placed before the current sampling window with
    negative local frame indices.  Unlike the original helper, this version
    does not silently reject the keyframes when the newer phase-0/absolute
    timeline scheduler makes ``(prev_tokens - steps) % 5 != 0``.  Every token
    retains its actual global H3 frame position through ``resolved_frame_index``.
    """
    n = next((g for g in (56, 39, 22, 5) if g <= int(n_frames)), 0)
    steps = steps_for_frames(n) if n else None
    if not steps or prev_video is None or steps > int(prev_tokens):
        return []
    start = int(prev_tokens) - int(steps)
    return [{
        "resolved_frame_index": frames_for_tokens(start + k) - int(f0),
        "latent": prev_video[:, :, start + k:start + k + 1].contiguous(),
    } for k in range(int(steps))]


def temporal_context_keyframes(prev_video, prev_audio, target_k0, overlap_tokens, motion_frames):
    """Build one contiguous fixed H3 context tail before the new target.

    temporal_overlap_frames contributes the nearest history. motion_anchor_frames
    contributes additional history immediately before that overlap context, matching
    the old node's adjacency while using proper non-updated H3 context rows.
    """
    if prev_video is None or target_k0 <= 0:
        return [], 0
    overlap_t = min(max(int(overlap_tokens), 0), target_k0)
    motion_t = _motion_context_tokens(int(motion_frames)) or 0
    motion_t = min(motion_t, max(0, target_k0 - overlap_t))
    total_t = overlap_t + motion_t
    if total_t <= 0:
        return [], 0

    start = target_k0 - total_t
    kfs = [{
        "kind": "context",
        "num_frames": total_t,
        "latent": prev_video[:, :, start:target_k0].contiguous(),
    }]

    if prev_audio is not None and prev_audio.shape[-1] > 0:
        audio_t = min(int(round(_context_span(total_t))), prev_audio.shape[-1])
        if audio_t > 0:
            kfs.append({
                "kind": "context_audio",
                "num_frames": audio_t,
                "audio_latent": prev_audio[:, :, :, -audio_t:].contiguous(),
            })
    return kfs, total_t

def warmup_keyframes(prev_video, prev_audio, sample_k0, fresh_k0, warmup_t):
    """Place previous refined tail at matching positions inside the new target.

    This mirrors tritant's Motion Context approach: each carried video token is
    an ordinary non-denoised H3 guide at the corresponding local pixel offset.
    The final overlap token is deliberately left without a video keyframe so
    the regenerated chunk has one H3 token to release from warm-up conditioning
    before entering its fresh suffix.  For the standard 22-frame / 7-token
    overlap this leaves the final 4 overlap frames as the release buffer.
    """
    if prev_video is None or warmup_t <= 0:
        return []
    if sample_k0 % 5 != 0:
        raise RuntimeError(
            f"H3 warm-up context must begin at phase 0, got token {sample_k0}."
        )
    if fresh_k0 - sample_k0 != warmup_t:
        raise RuntimeError("Warm-up token accounting mismatch.")

    tail = prev_video[:, :, sample_k0:fresh_k0]
    if tail.shape[2] != warmup_t:
        raise RuntimeError(
            f"Need {warmup_t} previous video tokens for warm-up, got {tail.shape[2]}."
        )

    kfs = []
    guided_t = max(0, warmup_t - 1)
    for j in range(guided_t):
        kfs.append({
            "resolved_frame_index": frames_for_tokens(j),
            "latent": tail[:, :, j:j + 1].contiguous(),
        })

    if prev_audio is not None and prev_audio.shape[-1] > 0:
        sf0 = frames_for_tokens(sample_k0)
        ff0 = frames_for_tokens(fresh_k0)
        a0, a1 = audio_range(sf0, ff0)
        a0 = max(0, min(a0, prev_audio.shape[-1]))
        a1 = max(a0, min(a1, prev_audio.shape[-1]))
        if a1 > a0:
            kfs.append({
                "resolved_frame_index": 0,
                "audio_latent": prev_audio[:, :, :, a0:a1].contiguous(),
            })
    return kfs

def identity_keyframes(source, conditioning_origin, search_start, f1, spacing, gate_strength):
    """Create phase-0 source identity keyframes outside regenerated overlap.

    ``conditioning_origin`` is the sampled chunk's frame-0 used for local
    resolved_frame_index coordinates. ``search_start`` is the earliest global
    source frame at which an identity anchor may be placed. For later temporal
    chunks this is the first fresh frame, so identity anchors never land inside
    the regenerated overlap prefix.
    """
    if spacing <= 0:
        return []
    cycle_frames = frames_for_tokens(5)  # 17 for H3's 1,4,4,4,4 cycle
    stride_cycles = max(1, int(round(float(spacing) / float(cycle_frames))))
    interval = stride_cycles * cycle_frames
    # Use the first phase-0 anchor strictly after search_start. This avoids
    # duplicating the boundary/overlap endpoint itself.
    first = ((int(search_start) // interval) + 1) * interval
    kfs, p = [], first
    while p < f1:
        cycle_index = p // cycle_frames
        k = cycle_index * 5
        if k < source.shape[2] and k % 5 == 0:
            kfs.append({
                "resolved_frame_index": int(p - conditioning_origin),
                "latent": source[:, :, k:k + 1].contiguous(),
                _MMH3_IDENTITY_GATE_KEY: max(0.0, min(1.0, float(gate_strength))),
            })
        p += interval
    if len(kfs) > MAX_IDENTITY_ANCHORS:
        kfs = kfs[:MAX_IDENTITY_ANCHORS]
    return kfs


def _merge_identity_keyframes(*groups):
    """Merge auto identity groups, preferring earlier groups at duplicate local frames."""
    out = []
    seen = set()
    for group in groups:
        for kf in group or ():
            idx = int(kf.get("resolved_frame_index", -10**9))
            if idx in seen:
                continue
            seen.add(idx)
            out.append(kf)
    return out


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    if lt.shape[3] == src_h and lt.shape[4] == src_w:
                        nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                    else:
                        lt_r = torch.nn.functional.interpolate(
                            lt.to(torch.float32), size=(src_h, src_w),
                            mode="bilinear", align_corners=False)
                        nkf["latent"] = lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out

# ---------------------------------------------------------------------------
# 空间网格
# ---------------------------------------------------------------------------
def _grid_1d(size, tile, ol, min_tile):
    if size <= tile:
        return [0], [size], [0]
    sh = tile - ol
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    rows = [i * sh for i in range(n)]
    trows = [min(tile, size - r) for r in rows]
    if min_tile > 0 and n >= 2:
        edge = size - rows[-1]
        if edge < min_tile:
            new_last = size - min_tile
            if rows[-2] < new_last < rows[-2] + trows[-2]:
                rows[-1] = new_last
                trows[-1] = size - new_last
    ovl = [0] * n
    for i in range(1, n):
        ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
    return rows, trows, ovl

def compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_th=0, min_tw=0):
    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl

def spatial_fade_mask(tile_h, tile_w, ovh, ovw, done_top, done_left,
                      fade_h=0, fade_w=0, seam_cap=1.0):
    """1=自由重采样, 0=冻结。
    重叠带 = 冻结段(缝侧) + 渐变段(0->seam_cap);
    seam_cap<1 时重叠带后再加一段等宽渐变 (cap->1), 让缝邻域以中等降噪
    "续写"冻结条内容, 高降噪下防止运动物体被切断。seam_cap=1.0 = 经典行为。"""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)

    def profile(n, ov, fade):
        p = torch.ones(n, dtype=torch.float32)
        f = min(fade, ov)
        frozen = ov - f
        p[:frozen] = 0.0
        if f > 0:
            p[frozen:ov] = torch.linspace(0.0, seam_cap, f)
        ramp = min(ov, n - ov)
        if ramp > 0 and seam_cap < 1.0:
            start = seam_cap if f > 0 else 0.0
            p[ov:ov + ramp] = torch.linspace(start, 1.0, ramp)
        return p

    if done_left and ovw > 0:
        mask = torch.minimum(mask, profile(tile_w, ovw, fade_w)[None, :])
    if done_top and ovh > 0:
        mask = torch.minimum(mask, profile(tile_h, ovh, fade_h)[:, None])
    return mask

# ---------------------------------------------------------------------------
# 校正 + 门控
# ---------------------------------------------------------------------------
def dc_correct(new, refs, clamp=COLOR_CLAMP, min_samples=256):
    pairs = [p for p in refs if p is not None and p[0] is not None
             and p[0].numel() >= min_samples]
    if not pairs:
        return new, None
    pa = torch.cat([a.float().permute(0, 2, 3, 4, 1).reshape(-1, a.shape[1]) for a, _ in pairs])
    pb = torch.cat([b.float().permute(0, 2, 3, 4, 1).reshape(-1, b.shape[1]) for _, b in pairs])
    dc = (pa - pb).median(dim=0).values.clamp(-clamp, clamp)
    return new - dc.view(1, -1, 1, 1, 1).to(new.device, new.dtype), dc

def grade_pin(chunk, ref, clamp=COLOR_CLAMP):
    a = chunk.float().permute(0, 2, 3, 4, 1).reshape(-1, chunk.shape[1])
    b = ref.float().permute(0, 2, 3, 4, 1).reshape(-1, ref.shape[1])
    dc = (a - b).median(dim=0).values.clamp(-clamp, clamp)
    return chunk - dc.view(1, -1, 1, 1, 1).to(chunk.device, chunk.dtype), dc

def seam_metrics(sub, region):
    if sub.numel() < 4096:
        return None, None
    sp = sub.float().permute(0, 2, 3, 4, 1).reshape(-1, sub.shape[1])
    rp = region.float().permute(0, 2, 3, 4, 1).reshape(-1, region.shape[1])
    dc = (sp - rp).median(dim=0).values.clamp(-DC_MATCH_CLAMP, DC_MATCH_CLAMP)
    a = sp - sp.mean(dim=0)
    b = rp - rp.mean(dim=0)
    corr = ((a * b).mean(dim=0) / (a.std(dim=0) * b.std(dim=0) + 1e-6)).median()
    return dc, float(corr)

def _should_polish(mode, sub, region):
    if mode == "all":
        return True
    dc, corr = seam_metrics(sub, region)
    if dc is None:
        return False
    return ((corr is not None and corr < SEAM_CORR_GATE)
            or dc.abs().max().item() > SEAM_DC_GATE * DC_MATCH_CLAMP)

# ---------------------------------------------------------------------------
# 时间缝合
# ---------------------------------------------------------------------------
def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * w.view(shape)

def temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0, f0, color_match=True):
    """Append a sampled temporal chunk, crossfading duplicate overlap latents.

    Both A and B independently sample the overlap.  The final assembled latent
    transitions from A to B across that duplicate region, then B remains fully
    authoritative for its fresh suffix.
    """
    if acc_v is None:
        return chunk_v, chunk_a
    gi, agi = k0, round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    rv = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                     device=acc_v.device, dtype=acc_v.dtype)
    ra = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    rv[:, :, :acc_v.shape[2]] = acc_v
    ra[:, :, :, :acc_a.shape[-1]] = acc_a
    v, a = chunk_v, chunk_a
    if index > 0:
        ov = max(0, min(acc_v.shape[2] - gi, v.shape[2]))
        if ov > 0:
            if color_match:
                v, dc = dc_correct(v, [(v[:, :, :ov], rv[:, :, gi:gi + ov])])
                if dc is not None:
                    print(f"[H3] 🎨 chunk {index} 时间颜色匹配 |dc|max={dc.abs().max():.4f}")
            rv[:, :, gi:gi + ov] = _crossfade(
                rv[:, :, gi:gi + ov], v[:, :, :ov], dim=2)
            v = v[:, :, ov:]
            gi += ov

        ova = max(0, min(acc_a.shape[-1] - agi, a.shape[-1]))
        if ova > 0:
            ra[:, :, :, agi:agi + ova] = _crossfade(
                ra[:, :, :, agi:agi + ova], a[:, :, :, :ova], dim=3)
            a = a[:, :, :, ova:]
            agi += ova

    if v.shape[2] > 0:
        rv[:, :, gi:gi + v.shape[2]] = v
    if a.shape[-1] > 0:
        ra[:, :, :, agi:agi + a.shape[-1]] = a
    return rv, ra

# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider

def sample_piece(piece, guider, noise_tensor, seed, sampler, sigmas, callback=None):
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model=guider.model_patcher.model, latent_image=latent_image)
    latent["samples"] = latent_image
    if callback is None:
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    samples = guider.sample(noise_tensor, latent_image, sampler, sigmas,
                            denoise_mask=latent.get("noise_mask"), callback=callback,
                            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())


def make_tile_progress(model_patcher, steps, n_tiles):
    previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)
    total = steps * n_tiles
    pbar = comfy.utils.ProgressBar(total)
    def for_tile(idx):
        def callback(step, x0, x, total_steps):
            preview = None
            if previewer is not None and x0 is not None:
                px0 = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                try:
                    preview = previewer.decode_latent_to_preview_image("JPEG", px0)
                except Exception:
                    preview = None
            pbar.update_absolute(idx * steps + step + 1, total, preview)
        return callback
    return for_tile

# ---------------------------------------------------------------------------
# 参数节点
# ---------------------------------------------------------------------------
class H3TemporalSplitParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalSplitParams", display_name="MMH3 Temporal Split Params",
            category="latent/upscale/minimax",
            description=("Temporal split controls. The overlap is regenerated and crossfaded, while previous refined output is supplied through original-style motion-history keyframes and matching warm-up keyframes. Total chunk window length stays fixed."),
            inputs=[
                io.Int.Input("chunk_frames", default=73, min=5, max=100000, step=1),
                io.Int.Input("temporal_overlap_frames", default=22, min=0, max=100000, step=1),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001,
                               tooltip="Single stock H3 visual-condition noise-aug applied to motion, warm-up and identity keyframes plus existing visual conditions in each split chunk."),
                io.Float.Input("identity_anchor_gate", default=0.30, min=0.0, max=1.0, step=0.01,
                               tooltip="Identity anchors are active while t_v < gate, equivalently sigma_v > 1-gate. 0.30 drops them once sigma_v <= 0.70. 1.0 keeps them for the full run."),
                io.Combo.Input("motion_anchor_frames", options=["0", "5", "22", "39"], default="22",
                               tooltip="Previous refined output supplied as negative-time H3 motion-history keyframes before each later sampling window."),
                io.Int.Input("identity_anchor_frames", default=24, min=0, max=240, step=1,
                             tooltip="Approximate identity-anchor interval. Snapped to H3 phase-0 one-frame positions on the 17-frame temporal cycle; 0 disables."),
                io.Boolean.Input("absolute_timeline_positions", default=True,
                                 tooltip="Give every split chunk its absolute H3 timeline/RoPE origin instead of restarting at local time zero."),
            ],
            outputs=[H3_TEMPORAL_PARAM.Output("temporal_split_param")],
        )

    @classmethod
    def execute(cls, chunk_frames, temporal_overlap_frames, anchor_strength,
                identity_anchor_gate, motion_anchor_frames, identity_anchor_frames,
                absolute_timeline_positions=True) -> io.NodeOutput:
        chunk = snap_clip_frames(int(chunk_frames))
        overlap = snap_overlap_frames(int(temporal_overlap_frames))
        if overlap >= chunk:
            overlap = snap_overlap_frames(chunk - 17)
        return io.NodeOutput({"p": (chunk, overlap, float(anchor_strength),
                                    float(identity_anchor_gate), int(motion_anchor_frames),
                                    int(identity_anchor_frames),
                                    bool(absolute_timeline_positions))})


class H3SpatialSplitParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SpatialSplitParams", display_name="MMH3 Spatial Split Params",
            category="latent/upscale/minimax",
            description="简化空间分块: 百分比重叠/渐变 + 缝邻域降噪上限。",
            inputs=[
                io.Int.Input("tile_width", default=512, min=64, max=16384, step=32),
                io.Int.Input("tile_height", default=512, min=64, max=16384, step=32),
                io.Float.Input("overlap_ratio", default=0.25, min=0.0, max=0.90, step=0.05),
                io.Float.Input("fade_ratio", default=0.50, min=0.0, max=1.0, step=0.05),
                io.Int.Input("min_tile_size", default=256, min=0, max=16384, step=32),
                io.Float.Input("seam_denoise", default=1.0, min=0.1, max=1.0, step=0.05,
                               tooltip="缝邻域降噪上限: <1 时高降噪下缝附近以中等降噪续写邻居内容, "
                                       "防止快运动物体在缝处被切断; 建议 0.5~0.8; 1.0=关。"),
            ],
            outputs=[
                H3_SPATIAL_PARAM.Output("spatial_split_param"),
                io.String.Output("grid_preview"),
            ],
        )

    @classmethod
    def execute(cls, tile_width, tile_height, overlap_ratio, fade_ratio,
                min_tile_size, seam_denoise) -> io.NodeOutput:
        tw, th = px_to_lat(tile_width), px_to_lat(tile_height)
        ol_w = min(tw - ALIGN, snap_align(tw * overlap_ratio))
        ol_h = min(th - ALIGN, snap_align(th * overlap_ratio))
        fw = min(ol_w, int(round(ol_w * fade_ratio)))
        fh = min(ol_h, int(round(ol_h * fade_ratio)))
        mt = min(px_to_lat(min_tile_size), th, tw) if min_tile_size > 0 else 0
        param = {"tw": tw, "th": th, "ol_w": ol_w, "ol_h": ol_h,
                 "fw": fw, "fh": fh, "mt": mt, "cap": float(seam_denoise)}
        preview = (f"Tile: {tile_width}x{tile_height}px -> {tw}x{th}lat | "
                   f"Overlap: {overlap_ratio:.0%} -> {ol_w}x{ol_h}lat | "
                   f"Fade: {fade_ratio:.0%} -> {fw}x{fh}lat | SeamDenoise: {seam_denoise:.2f}")
        print(f"[H3] 📊 {preview}")
        return io.NodeOutput(param, preview)


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------
class MMH3SplitUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SplitUpscale", display_name="MMH3 Split Upscale",
            category="latent/upscale/minimax",
            description=("H3 split latent upscale with absolute timeline positioning and regenerated temporal-overlap latent crossfade, configured by MMH3 Temporal Split Params."),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                H3_TEMPORAL_PARAM.Input("temporal_split_param", optional=True),
                H3_SPATIAL_PARAM.Input("spatial_split_param", optional=True),
                io.Combo.Input("seam_polish", options=["off", "auto", "all"], default="off"),
                io.Boolean.Input("color_match", default=True),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, latent, conditioning, model, noise, sampler, sigmas,
                negative=None, cfg=1.0, temporal_split_param=None,
                spatial_split_param=None, seam_polish="off",
                color_match=True) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("期望 MiniMax H3 AV latent (嵌套 video+audio)")
        video, audio = samples.tensors[0], samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("仅支持 Batch 1")
        B, C, T, H, W = video.shape

        if temporal_split_param is not None:
            _tp = temporal_split_param["p"]
            if len(_tp) < 7:
                raise ValueError("MMH3 Temporal Split Params is from an older node revision; recreate that params node once.")
            (cl, ov, anchor_strength, identity_anchor_gate, motion_n, identity_n,
             absolute_timeline_positions) = _tp[:7]
            bounds, _ = compute_h3_segments_adaptive(T, cl, ov)
            overlap_tokens = clip_tokens(ov) if ov > 0 else 0
            total_tokens = clip_tokens(cl)
            hop_tokens = max(1, total_tokens - overlap_tokens) if overlap_tokens > 0 else total_tokens
            print(f"[H3] temporal overlap mode: window={total_tokens} tokens, "
                  f"regenerated-overlap={overlap_tokens}, fresh-hop={hop_tokens}")
            print(f"[H3] anchor strength={anchor_strength:.3f}; identity gate={identity_anchor_gate:.3f} "
                  f"(drop at sigma_v <= {1.0 - identity_anchor_gate:.3f})")
            print(f"[H3] previous-output motion anchors: {motion_n} frame(s)")
        else:
            bounds = [(0, 0, T, 0)]
            anchor_strength, identity_anchor_gate, motion_n, identity_n = 0.999, 1.0, 0, 0
            absolute_timeline_positions = False
            overlap_tokens = 0

        if spatial_split_param is not None:
            sp = spatial_split_param
            rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(
                H, W, sp["th"], sp["tw"], sp["ol_h"], sp["ol_w"], sp["mt"], sp["mt"])
            seam_cap = sp.get("cap", 1.0)
        else:
            rows, cols, trows, tcols, row_ovl, col_ovl = [0], [0], [H], [W], [0], [0]
            seam_cap = 1.0
        nrows, ncols = len(rows), len(cols)

        use_absolute_timeline = bool(absolute_timeline_positions) and temporal_split_param is not None
        if use_absolute_timeline:
            if not _MMH3_ABSOLUTE_TIMELINE_PATCH_OK:
                raise RuntimeError(
                    "absolute_timeline_positions is enabled but the H3 PackedLayout patch is unavailable")
            print("[H3] absolute timeline positions: ON (chunk RoPE origin follows global frame offset)")
        elif bool(absolute_timeline_positions):
            print("[H3] absolute timeline positions: no temporal split; window origin remains 0")
        else:
            print("[H3] absolute timeline positions: OFF (stock local-window target positions)")

        print(f"[H3] regenerated latent overlap + crossfade: {'ON (' + str(overlap_tokens) + ' token(s))' if overlap_tokens > 0 else 'OFF'}")

        steps = max(int(sigmas.shape[-1]) - 1, 1)
        n_tiles = len(bounds) * nrows * ncols
        for_tile = make_tile_progress(model.model_patcher if hasattr(model, "model_patcher") else model,
                                      steps, n_tiles)

        source = video
        noise_v = noise.generate_noise({"samples": torch.zeros_like(video, dtype=torch.float32)})
        noise_a = noise.generate_noise({"samples": torch.zeros_like(audio, dtype=torch.float32)})

        acc_v = acc_a = None
        polish_queue = {}
        tile_idx = 0

        for i, (sample_k0, fresh_k0, fresh_k1, warmup_t) in enumerate(bounds):
            # The overlap remains inside the fixed-size H3 window and is sampled
            # normally by B. Final assembly later crossfades A/B across those
            # duplicate overlap tokens, so no temporal denoise mask is required.
            overlap_t = int(warmup_t) if i > 0 else 0

            sample_f0 = frames_for_tokens(sample_k0)
            if use_absolute_timeline:
                print(f"[H3] chunk {i}: absolute window start={sample_f0} frames "
                      f"(RoPE time +{float(FRAME_RESCALE) * float(sample_f0):.3f})")
            fresh_f0 = frames_for_tokens(fresh_k0)
            fresh_f1 = frames_for_tokens(fresh_k1)
            warmup_frames = fresh_f0 - sample_f0

            sample_k1 = fresh_k1
            sample_f1 = frames_for_tokens(sample_k1)

            # Sample the full overlapping window from the input/upscaled latent.
            # Unlike the frozen-prefix experiment, A's previous HR result is NOT
            # copied into B here: B produces its own overlap trajectory.
            source_chunk = video[:, :, sample_k0:sample_k1].contiguous()
            overlap_active = bool(overlap_t > 0 and i > 0)
            if overlap_active:
                print(f"[H3] chunk {i}: regenerate overlap tokens "
                      f"{sample_k0}:{fresh_k0} ({overlap_t} tokens / {warmup_frames} frames)")

            a0, a1 = audio_range(sample_f0, sample_f1)
            a1 = min(a1, audio.shape[-1])
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            # User conditioning may extend into the sampled look-ahead, but the
            # auto-generated identity anchors below still stop at nominal fresh_f1.
            cond_i = reanchor_conditioning(conditioning, sample_f0, sample_f1, (H, W))

            # Restore the original repository behavior: provide a trailing
            # previous-output motion history immediately before this sampling
            # window as ordinary negative-time H3 keyframes.
            if i > 0 and acc_v is not None and motion_n > 0:
                motion_ids = motion_keyframes(
                    acc_v, sample_k0, sample_f0, motion_n)
                if motion_ids:
                    cond_i = prepend_keyframes(cond_i, motion_ids)
                    print(f"[H3] chunk {i}: previous-output motion context: "
                          f"{len(motion_ids)} token(s) / requested {motion_n} frames")

            # Guide B's regenerated overlap with A's actual refined overlap at
            # the corresponding local H3 positions.  These are conditioning
            # rows, not a frozen target prefix; B remains free to reconcile the
            # overlap with its fresh future frames.
            if overlap_active and acc_v is not None:
                warm_ids = warmup_keyframes(
                    acc_v, acc_a, sample_k0, fresh_k0, overlap_t)
                if warm_ids:
                    cond_i = prepend_keyframes(cond_i, warm_ids)
                    warm_video = sum(1 for kf in warm_ids if kf.get("latent") is not None)
                    print(f"[H3] chunk {i}: refined overlap warm-up: "
                          f"{warm_video} video token(s)")

            periodic_ids = []
            if identity_n > 0:
                identity_search_start = fresh_f0 if overlap_active else sample_f0
                periodic_ids = identity_keyframes(
                    source, sample_f0, identity_search_start, fresh_f1, identity_n,
                    identity_anchor_gate)

            auto_ids = _merge_identity_keyframes(periodic_ids)
            if auto_ids:
                cond_i = prepend_keyframes(cond_i, auto_ids)

            # Stock ComfyUI H3 uses one visual conditioning strength for all
            # keyframes/refs in this chunk. No runtime source patching is needed.
            cond_i = set_anchor_strength(cond_i, anchor_strength)

            # One owned full-chunk buffer is enough.  The old path first made
            # source_chunk contiguous as chunk_v and then cloned it again into
            # chunk_out, retaining two full extended temporal chunks.
            chunk_out = source_chunk.contiguous()
            noise_vc = noise_v[:, :, sample_k0:sample_k1]
            noise_ac = noise_a[:, :, :, a0:a1]


            # ================= 空间内循环 =================
            for ri in range(nrows):
                for cj in range(ncols):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    r0, c0 = rows[ri], cols[cj]
                    tr, tc = trows[ri], tcols[cj]
                    ovh, ovw = row_ovl[ri], col_ovl[cj]

                    tile = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                    if spatial_split_param is not None:
                        fh, fw = spatial_split_param["fh"], spatial_split_param["fw"]
                    else:
                        fh = fw = 0
                    m = spatial_fade_mask(tr, tc, ovh, ovw,
                                          done_top=(ri > 0), done_left=(cj > 0),
                                          fade_h=fh, fade_w=fw, seam_cap=seam_cap)
                    # Temporal overlap is regenerated normally; the denoise mask
                    # here is spatial-only. Temporal blending happens after sampling.
                    mv = m[None, None, None].to(chunk_out.device)
                    ma = torch.zeros((1, 32, 2, chunk_a.shape[-1]),
                                     device=chunk_a.device, dtype=torch.float32)
                    piece = {"samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                             "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma))}
                    tile_noise = comfy.nested_tensor.NestedTensor((
                        noise_vc[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous(),
                        noise_ac.contiguous()))

                    cond_tile = crop_keyframes_to_tile(cond_i, H, W, r0, c0, tr, tc)
                    guider = build_guider(model, cond_tile, negative, cfg)
                    window_start_frames = float(sample_f0) if use_absolute_timeline else 0.0
                    window_start_token = int(sample_k0) if use_absolute_timeline else 0
                    _timeline_frame_ctx = _MMH3_WINDOW_START_FRAMES.set(window_start_frames)
                    _timeline_token_ctx = _MMH3_WINDOW_START_TOKEN.set(window_start_token)
                    try:
                        out = sample_piece(
                            piece, guider, tile_noise, noise.seed, sampler, sigmas,
                            callback=for_tile(tile_idx))
                    finally:
                        _MMH3_WINDOW_START_TOKEN.reset(_timeline_token_ctx)
                        _MMH3_WINDOW_START_FRAMES.reset(_timeline_frame_ctx)
                    tile_v = (out.tensors[0] if out.is_nested else out).to(chunk_out.device)

                    region = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                    if color_match:
                        tile_v, _ = dc_correct(tile_v, [
                            (tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw])
                                if (cj > 0 and ovw > 0) else None,
                            (tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :])
                                if (ri > 0 and ovh > 0) else None,
                            (tile_v, source_chunk[:, :, :, r0:r0 + tr, c0:c0 + tc])
                                if (ri == 0 and cj == 0) else None,
                        ])

                    if seam_polish != "off":
                        if cj > 0 and ovw > 0 and \
                                _should_polish(seam_polish, tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw]):
                            polish_queue[(i, ri, cj, "W")] = (c0, ovw, (sample_k0, fresh_k1))
                        if ri > 0 and ovh > 0 and \
                                _should_polish(seam_polish, tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :]):
                            polish_queue[(i, ri, cj, "H")] = (r0, ovh, (sample_k0, fresh_k1))

                    if cj > 0 and ovw > 0:
                        wts = torch.linspace(0.0, 1.0, ovw, device=region.device,
                                             dtype=region.dtype).view(1, 1, 1, 1, ovw)
                        region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts)
                                                    + tile_v[:, :, :, :, :ovw] * wts)
                    if ri > 0 and ovh > 0:
                        wts = torch.linspace(0.0, 1.0, ovh, device=region.device,
                                             dtype=region.dtype).view(1, 1, 1, ovh, 1)
                        region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts)
                                                    + tile_v[:, :, :, :ovh, :] * wts)
                    band = torch.zeros((1, 1, 1, tr, tc), dtype=torch.bool, device=region.device)
                    if cj > 0 and ovw > 0:
                        band[:, :, :, :, :ovw] = True
                    if ri > 0 and ovh > 0:
                        band[:, :, :, :ovh, :] = True
                    region = torch.where(band, region, tile_v)
                    chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

                    # Drop tile-sized sampler outputs and helper tensors immediately.
                    # In particular, do not retain the stage-2 output plus handoff slice
                    # while the next spatial tile starts allocating.
                    del region, band, tile_v, out
                    del guider, cond_tile, tile_noise, piece, ma, mv, m, tile

                    tile_idx += 1


            if color_match:
                chunk_out, dcg = grade_pin(chunk_out, source_chunk)
                if dcg is not None and dcg.abs().max().item() > 1e-4:
                    print(f"[H3] 🎯 chunk {i} 全局钉源 |dc|max={dcg.abs().max():.4f}")

            # ================= 修复版 polish 二道缝 =================
            chunk_polish = [(k, v) for k, v in polish_queue.items() if k[0] == i]
            if chunk_polish:
                print(f"[H3] 🔧 chunk {i}: polish {len(chunk_polish)} 条缝")
                pbar2 = comfy.utils.ProgressBar(steps * len(chunk_polish))
                for pi, (key, (s0, band_w, (tk0, tk1))) in enumerate(chunk_polish):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    axis = key[3]
                    t_len = tk1 - tk0
                    if axis == "W":
                        w0 = max(0, s0 - POLISH_HALO)
                        w1 = min(W, s0 + band_w + POLISH_HALO)
                        win = chunk_out[:, :, :, :, w0:w1].clone()
                        b0, b1 = s0 - w0, s0 - w0 + band_w
                        mv = torch.zeros((1, 1, t_len, H, w1 - w0), dtype=torch.float32, device=win.device)
                        mv[:, :, :, :, b0:b1] = 1.0
                        r0c, c0c, trc, tcc, ax = 0, w0, H, w1 - w0, 4
                        nsl = (slice(None), slice(None), slice(tk0, tk1), slice(None), slice(w0, w1))
                    else:
                        h0 = max(0, s0 - POLISH_HALO)
                        h1 = min(H, s0 + band_w + POLISH_HALO)
                        win = chunk_out[:, :, :, h0:h1, :].clone()
                        b0, b1 = s0 - h0, s0 - h0 + band_w
                        mv = torch.zeros((1, 1, t_len, h1 - h0, W), dtype=torch.float32, device=win.device)
                        mv[:, :, :, b0:b1, :] = 1.0
                        r0c, c0c, trc, tcc, ax = h0, 0, h1 - h0, W, 3
                        nsl = (slice(None), slice(None), slice(tk0, tk1), slice(h0, h1), slice(None))
                    ma = torch.zeros((1, 32, 2, chunk_a.shape[-1]),
                                     device=chunk_a.device, dtype=torch.float32)
                    piece = {"samples": comfy.nested_tensor.NestedTensor((win, chunk_a)),
                             "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma))}
                    tn = comfy.nested_tensor.NestedTensor((noise_v[nsl].contiguous(), noise_ac.contiguous()))

                    cond_p = crop_keyframes_to_tile(cond_i, H, W, r0c, c0c, trc, tcc)
                    g = build_guider(model, cond_p, negative, cfg)

                    def _cb(step, x0, x, ts, _pi=pi):
                        pbar2.update_absolute(_pi * steps + step + 1, steps * len(chunk_polish))

                    polish_window_start = float(frames_for_tokens(tk0)) if use_absolute_timeline else 0.0
                    polish_window_token = int(tk0) if use_absolute_timeline else 0
                    _polish_frame_ctx = _MMH3_WINDOW_START_FRAMES.set(polish_window_start)
                    _polish_token_ctx = _MMH3_WINDOW_START_TOKEN.set(polish_window_token)
                    try:
                        out = sample_piece(piece, g, tn, noise.seed, sampler, sigmas, callback=_cb)
                    finally:
                        _MMH3_WINDOW_START_TOKEN.reset(_polish_token_ctx)
                        _MMH3_WINDOW_START_FRAMES.reset(_polish_frame_ctx)
                    pv = (out.tensors[0] if out.is_nested else out).to(chunk_out.device)

                    nlen = (w1 - w0) if ax == 4 else (h1 - h0)
                    alpha = torch.ones(nlen, dtype=torch.float32)
                    if b0 > 0:
                        alpha[:b0] = torch.linspace(0.0, 1.0, b0)
                    if nlen - b1 > 0:
                        alpha[b1:] = torch.linspace(1.0, 0.0, nlen - b1)
                    view = [1, 1, 1, 1, 1]
                    view[ax] = nlen
                    avv = alpha.view(view).to(chunk_out.device)
                    if ax == 4:
                        chunk_out[:, :, :, :, w0:w1] = avv * pv + (1.0 - avv) * win
                    else:
                        chunk_out[:, :, :, h0:h1, :] = avv * pv + (1.0 - avv) * win
                    del pv, out, g, cond_p, tn, piece, ma, mv, win


            # Stitch the COMPLETE sampled window. For later chunks this includes
            # B's regenerated overlap; temporal_append crossfades the duplicate
            # A/B latent region and then appends B's fresh suffix.
            acc_v, acc_a = temporal_append(
                acc_v, acc_a, chunk_out, chunk_a, i, sample_k0, sample_f0,
                color_match=False)

            # End-of-chunk release. acc_v/acc_a are the only temporal tensors
            # intentionally kept for the next iteration.
            del chunk_out, chunk_a, source_chunk, cond_i
            del noise_vc, noise_ac
            # A chunk boundary is a useful allocator reset point on low-VRAM GPUs.
            comfy.model_management.soft_empty_cache()

        # The full source/noise tensors are no longer needed once all chunks have
        # been assembled.  Releasing these references before model unload gives
        # the following VAE decode as much CPU/GPU/offload headroom as possible.
        del source, noise_v, noise_a, video, audio, samples
        gc.collect()

        # Release the diffusion model and its attached auxiliary models before
        # the downstream VAE starts AIMDO file-backed weight streaming. This is
        # specifically to maximize Windows host/virtual-memory headroom and
        # reduce error 1450 failures during VAE decode.
        try:
            comfy.model_management.unload_model_and_clones(
                model, unload_additional_models=True)
        finally:
            gc.collect()
            comfy.model_management.soft_empty_cache()

        # Final handoff to stock VAE decode. acc_v/acc_a are intentionally the
        # only large tensors still alive here.
        final_samples = comfy.nested_tensor.NestedTensor((acc_v, acc_a))
        return io.NodeOutput({"samples": final_samples})



NODE_CLASS_MAPPINGS = {
    "MMH3TemporalSplitParams": H3TemporalSplitParams,
    "MMH3SpatialSplitParams": H3SpatialSplitParams,
    "MMH3SplitUpscale": MMH3SplitUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3TemporalSplitParams": "MMH3 Temporal Split Params",
    "MMH3SpatialSplitParams": "MMH3 Spatial Split Params",
    "MMH3SplitUpscale": "MMH3 Split Upscale",
}
