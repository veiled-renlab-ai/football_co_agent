"""Per-frame post-processing for the gfootball MJPEG stream.

Pipeline:
  1. remove_hud      — zero gfootball's built-in scoreboard strip
  2. grade           — S-curve on L channel + saturation + warm shift
  3. film_grain      — subtle luminance noise; breaks flat CG surfaces
  4. highlight_bloom — soft halo on bright areas; mimics real camera lens
  5. sharpen         — mild unsharp-mask
  6. vignette        — gentle dark-corner vignette

Why grain + bloom?
  gfootball uses flat-shaded low-poly models (basic OpenGL, no PBR).  Film grain
  adds micro-texture to uniform colour patches, tricking the eye into perceiving
  surface detail.  Highlight bloom makes the image feel camera-captured rather
  than CG-rendered.  Together they remove the "cheap game asset" look more
  effectively than any colour grade alone.

Tuning constants (edit freely):
  HUD_H_FRAC     fraction of frame height to black out
  GAMMA          < 1 brightens; 0.82 gives a +20% midtone lift
  SAT_BOOST      colour richness multiplier (1.0 = none)
  WARM_R / B     RGB warm shift (additive, [0-255])
  GRAIN_SIGMA    film-grain strength; 3–5 = subtle, 8+ = visible
  BLOOM_THRESH   luminance threshold above which bloom applies [0-255]
  BLOOM_SIGMA    gaussian spread of bloom (pixels at full resolution)
  BLOOM_STR      bloom intensity [0-1]; 0.18 = barely visible
  SHARP_AMT      unsharp-mask strength
  VIG_INTENSITY  0 = no vignette, 0.22 = light
"""
from __future__ import annotations

import numpy as np
import cv2

# ── Tuning ────────────────────────────────────────────────────────────────────
HUD_H_FRAC:   float = 0.09
GAMMA:        float = 0.82   # midtone brightening (0.82 ≈ +20%)
SAT_BOOST:    float = 1.20
WARM_R:       float = 6.0
WARM_B:       float = 3.0
GRAIN_SIGMA:  float = 0.0    # film grain — OFF (set >0 to enable)
BLOOM_THRESH: int   = 220    # pixels brighter than this glow
BLOOM_SIGMA:  float = 22.0   # bloom spread (wide + soft)
BLOOM_STR:    float = 0.0    # bloom — OFF (set >0 to enable)
SHARP_AMT:    float = 0.22
SHARP_SIGMA:  float = 1.1
VIG_INTENSITY: float = 0.20

# ── Caches ────────────────────────────────────────────────────────────────────
_VIG_CACHE: dict[tuple, np.ndarray] = {}
_L_LUT:     np.ndarray | None = None


def _get_l_lut() -> np.ndarray:
    """S-curve LUT on the L channel: gamma midtone boost + sin shoulder."""
    global _L_LUT
    if _L_LUT is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        y = np.power(np.clip(x, 1e-6, 1.0), GAMMA)
        # Gentle S-shoulder: push midtones up, roll off highlights slightly.
        y = y + 0.05 * np.sin(np.pi * y)
        y = np.clip(y, 0.0, 1.0)
        _L_LUT = (y * 255).astype(np.uint8)
    return _L_LUT


# ── Entry point ───────────────────────────────────────────────────────────────

def process(bgr: np.ndarray) -> np.ndarray:
    """Full FX pipeline.  Input / output: uint8 BGR ndarray."""
    try:
        _remove_hud(bgr)
        out = _grade(bgr)
        out = _sharpen(out)
        return _vignette(out)
    except Exception:
        return bgr


# ── Steps ─────────────────────────────────────────────────────────────────────

def _remove_hud(bgr: np.ndarray) -> None:
    cut = max(1, int(bgr.shape[0] * HUD_H_FRAC))
    bgr[:cut] = 0


def _grade(bgr: np.ndarray) -> np.ndarray:
    # Luminance S-curve via LUT (no tile artifacts).
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _get_l_lut()[lab[:, :, 0]]
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Saturation boost.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SAT_BOOST, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Warm colour temperature.
    f = out.astype(np.float32)
    f[:, :, 2] = np.clip(f[:, :, 2] + WARM_R, 0, 255)
    f[:, :, 0] = np.clip(f[:, :, 0] - WARM_B, 0, 255)
    return f.astype(np.uint8)


def _film_grain(bgr: np.ndarray) -> np.ndarray:
    """Add subtle luminance noise — breaks flat CG surfaces, adds perceived depth."""
    h, w = bgr.shape[:2]
    # Generate at ½ res, upscale: smoother grain clump, 4× cheaper.
    noise_s = np.random.normal(0.0, GRAIN_SIGMA, (h // 2, w // 2)).astype(np.float32)
    noise = cv2.resize(noise_s, (w, h), interpolation=cv2.INTER_LINEAR)
    f = bgr.astype(np.float32) + noise[:, :, None]  # same noise on all ch → no hue shift
    return np.clip(f, 0, 255).astype(np.uint8)


def _highlight_bloom(bgr: np.ndarray) -> np.ndarray:
    """Soft halo on bright pixels — mimics real camera lens flare / overexposure."""
    h, w = bgr.shape[:2]
    # Work at ¼ resolution (bloom is wide; subpixel accuracy irrelevant).
    small = cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Extract only the brightest areas (soft threshold).
    bright = np.clip(gray - BLOOM_THRESH, 0, 255) / (255.0 - BLOOM_THRESH)
    bloom_s = cv2.GaussianBlur(bright, (0, 0), BLOOM_SIGMA / 4.0)
    bloom = cv2.resize(bloom_s, (w, h), interpolation=cv2.INTER_LINEAR)
    f = bgr.astype(np.float32) + bloom[:, :, None] * BLOOM_STR * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), SHARP_SIGMA)
    return cv2.addWeighted(bgr, 1.0 + SHARP_AMT, blur, -SHARP_AMT, 0)


def _vignette(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    key = (h, w)
    if key not in _VIG_CACHE:
        yv = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xv = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        dist = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(2.0)
        mask = np.clip(1.0 - dist * VIG_INTENSITY, 1.0 - VIG_INTENSITY, 1.0)
        _VIG_CACHE[key] = mask[:, :, None].astype(np.float32)
    return (bgr.astype(np.float32) * _VIG_CACHE[key]).astype(np.uint8)
