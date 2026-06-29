"""
features.py
-----------
Turns one image into a small vector of numbers ("features") that capture the
visual differences between a REAL photo and a PHOTO OF A SCREEN (recapture).

Why these features (the physics):

1. FREQUENCY-DOMAIN ENERGY (FFT)
   A screen has a regular sub-pixel grid. When you photograph it, that grid
   beats against the camera's own sensor grid and produces a moire pattern -
   a repeating texture that shows up as extra energy at mid/high spatial
   frequencies in the 2D Fourier transform. Real-world textures (skin, cloth,
   walls, leaves) don't have this regular grid, so their frequency energy
   falls off smoothly. We measure how much energy sits in mid/high bands
   relative to total energy, and how "spiky" (peaky) that energy is.

2. COLOR STATISTICS
   Screens emit light (backlight + RGB sub-pixels) instead of reflecting it.
   This tends to push colors towards a cooler/bluer cast, compress contrast
   in highlights, and produce a narrower, more uniform saturation range.
   We measure mean/std of hue, saturation, and the blue-vs-red balance.

3. LOCAL SHARPNESS / BLUR CONSISTENCY
   Real-world scenes have depth: parts of the image are sharp, parts are
   naturally out of focus (depth of field). A recapture is a flat 2D surface
   photographed by a camera, so its sharpness pattern is much more uniform
   across the frame. We tile the image and measure the variance of a
   Laplacian-based sharpness score across tiles - real photos vary more.

4. GLARE / SPECULAR HIGHLIGHTS
   Screens (and printouts under any indoor light) often have flat, bright,
   low-saturation patches caused by glare reflecting off a flat surface.
   We measure how much of the image is in "near-white, low-saturation,
   high-brightness" territory.

5. EDGE / BEZEL HINTS
   If a screen edge, bezel, or rectangular frame is visible, there's often a
   strong straight-line signature near the image border. We do a cheap check
   for strong horizontal/vertical gradient concentration near the border.

All features are scale-invariant-ish (images are resized to a fixed size
first) and computed with plain numpy/scipy/PIL so they're cheap and easy to
port to a phone (no heavy deep-learning runtime needed).
"""

import numpy as np
from PIL import Image
from scipy import ndimage

# Fixed working size keeps features comparable across different camera
# resolutions, and keeps everything fast.
WORK_SIZE = 512


def _load_rgb(image_path_or_img):
    if isinstance(image_path_or_img, Image.Image):
        img = image_path_or_img.convert("RGB")
    else:
        img = Image.open(image_path_or_img).convert("RGB")
    # Resize (don't crop) so every image is comparable & fast to process.
    img = img.resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR)
    return np.asarray(img).astype(np.float32)  # H x W x 3, 0-255


def _to_gray(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def _fft_features(gray):
    """Energy distribution in the frequency domain (moire / screen-grid signal)."""
    # Window to reduce edge-of-image artifacts in the FFT
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fft2(gray * win)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()

    # Ring masks: low / mid / high frequency bands
    low_mask = r_norm < 0.08
    mid_mask = (r_norm >= 0.08) & (r_norm < 0.30)
    high_mask = r_norm >= 0.30

    total = mag_log.sum() + 1e-8
    low_e = mag_log[low_mask].sum() / total
    mid_e = mag_log[mid_mask].sum() / total
    high_e = mag_log[high_mask].sum() / total

    # "Peakiness": screens often create a few sharp narrow spikes in the
    # mid band (moire), rather than smoothly distributed energy.
    mid_vals = mag_log[mid_mask]
    mid_peakiness = (mid_vals.max() / (mid_vals.mean() + 1e-8)) if mid_vals.size else 0.0

    return {
        "fft_low_e": low_e,
        "fft_mid_e": mid_e,
        "fft_high_e": high_e,
        "fft_mid_peakiness": mid_peakiness,
    }


def _color_features(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn

    sat = np.where(mx > 0, diff / (mx + 1e-8), 0.0)
    val = mx / 255.0

    blue_red_balance = (b.mean() - r.mean()) / 255.0  # screens often skew cool/blue

    return {
        "sat_mean": sat.mean(),
        "sat_std": sat.std(),
        "val_mean": val.mean(),
        "blue_red_balance": blue_red_balance,
        "r_std": r.std() / 255.0,
        "g_std": g.std() / 255.0,
        "b_std": b.std() / 255.0,
    }


def _sharpness_tiles(gray, tiles=8):
    """Variance of local sharpness across a tiled grid (depth-of-field cue)."""
    h, w = gray.shape
    th, tw = h // tiles, w // tiles
    lap = ndimage.laplace(gray)
    scores = []
    for i in range(tiles):
        for j in range(tiles):
            tile = lap[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if tile.size:
                scores.append(tile.var())
    scores = np.array(scores) if scores else np.array([0.0])
    return {
        "sharp_tile_mean": scores.mean(),
        "sharp_tile_std": scores.std(),
        "sharp_tile_cv": scores.std() / (scores.mean() + 1e-8),  # coefficient of variation
    }


def _glare_features(rgb):
    """Fraction of near-white, low-saturation, bright pixels (glare/screen glow)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0)
    bright = mx / 255.0

    glare_mask = (bright > 0.85) & (sat < 0.15)
    return {
        "glare_frac": glare_mask.mean(),
    }


def _border_edge_features(gray):
    """Cheap bezel/frame cue: strong straight gradients concentrated near border."""
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    h, w = grad_mag.shape
    border = max(2, int(0.04 * min(h, w)))

    border_mask = np.zeros_like(grad_mag, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    border_energy = grad_mag[border_mask].mean()
    interior_energy = grad_mag[~border_mask].mean() + 1e-8
    return {
        "border_edge_ratio": border_energy / interior_energy,
    }


FEATURE_NAMES = [
    "fft_low_e", "fft_mid_e", "fft_high_e", "fft_mid_peakiness",
    "sat_mean", "sat_std", "val_mean", "blue_red_balance",
    "r_std", "g_std", "b_std",
    "sharp_tile_mean", "sharp_tile_std", "sharp_tile_cv",
    "glare_frac",
    "border_edge_ratio",
]


def extract_features(image_path_or_img):
    """Main entry point: image -> fixed-length numpy feature vector."""
    rgb = _load_rgb(image_path_or_img)
    gray = _to_gray(rgb)

    feats = {}
    feats.update(_fft_features(gray))
    feats.update(_color_features(rgb))
    feats.update(_sharpness_tiles(gray))
    feats.update(_glare_features(rgb))
    feats.update(_border_edge_features(gray))

    vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, feats
