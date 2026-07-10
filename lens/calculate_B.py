"""Inverse of lens_reflection.py: recover the mask B from the scene A and result C.

Forward model (see lens_reflection.py):

    C = A * blur(B)              # blur() = convolution with the defocus PSF

So given A and C we can recover the (out-of-focus) mask shadow, then deconvolve
the PSF to get the sharp mask B:

    0.  C' = darken(C)           # pull C down so it is never brighter than A (C' in [0,1])
    1.  T_blur = C' / A          # the blurred transmittance the sensor actually saw
    2.  T      = deconvolve(T_blur, PSF)   # undo the defocus blur -> sharp mask
    3.  B      = T mapped to black/gray/white, resized to b1 x b2

Step 0 darkens C (multiplicatively) so a valid B (transmittance <= 1) exists;
step 1 is per-pixel division; step 2 is a Wiener deconvolution. C is assumed to
have the same resolution as A; B has b1 x b2.

The PSF (shape + radius) MUST match the one used in the forward pass, so this
file reuses build_psf / physical_blur_radius from lens_reflection.

Math formulas used in this file  (grep "FORMULA:" to jump between them)
-----------------------------------------------------------------------
FFT = 2-D Fourier transform, conj = complex conjugate, |.| = magnitude.

  FORMULA: Black floor on A  (avoid a 0 that no darkening of C can satisfy)
    does: lift pure-black (0,0,0) pixels of A to a near-black gray (1,1,1)=1/255
    A[pixel < 0.5/255] = 1/255                            [in darken_C_to_valid]

  FORMULA: Validity darkening  (global exposure-down so a transmittance B<=1 exists)
    does: darken C if A is darker than C anywhere (else no valid B)
    d = min( 1, min over pixels (L(C)>0) of L(A)/L(C) ),  C' = C*d  [in darken_C_to_valid]
    stops_down = log2(d)         (d <= 1, so C' stays in [0,1])

  FORMULA: Transmittance recovery  (inverse of the Multiply blend)
    does: undo "C' = A * T_blur" to get back the blurred mask the sensor saw
    T_blur = L(C') / L(A)       L = luma (Rec.601)        [in estimate_blurred_transmittance]
    (C' <= A everywhere, so T_blur is already in [0,1])

  FORMULA: PSF -> OTF  (Optical Transfer Function = FFT of the PSF)
    does: move the blur kernel into the frequency domain at full image size
    H = FFT(psf)                                          [in _psf2otf]

  FORMULA: Wiener deconvolution  (a.k.a. Wiener filter / regularized inverse filter)
    does: remove the defocus blur to recover the SHARP mask (the hard step)
    G = FFT(T_blur)
    F = G * conj(H) / ( |H|^2 + NSR )                     [in deconvolve]
    T = Re( IFFT(F) )           NSR = noise_ratio
    note: with NSR=0 this is plain inverse filtering F = G/H; NSR regularizes
          the division wherever H is near zero (prevents noise blow-up / ringing).

  FORMULA: Recovered mask
    does: clip to valid range and resize to the requested mask size
    B = clip(T, 0, 1)  ->  resize to (b1, b2)            [in recover_mask]
"""

import numpy as np
from PIL import Image

from lens_reflection import (
    read_image,
    resize_image,
    normalize_transparency,
    build_psf,
    physical_blur_radius,
)


# ---------------------------------------------------------------------------
# 1. darken C so A is never darker than C, so a valid mask B (B<=1) exists
# ---------------------------------------------------------------------------
def darken_C_to_valid(scene, result):
    """Darken C (multiplicatively) so A is never darker than C at any pixel.

    The forward model is C = A * B with transmittance B in [0, 1], so C can never be
    brighter than A. If a (real / noisy) A is darker than C somewhere, no valid B
    exists there. Instead of brightening A, we DARKEN C by a single global exposure
    gain d (<= 1) -- the largest gain that pulls C down to <= A everywhere.

    FORMULA: Black floor on A -- lift pure-black (0,0,0) pixels of A to gray (1,1,1)=1/255
      A[pixel < 0.5/255] = 1/255
      ("(0,0,0)" = 8-bit black; a black A pixel must be floored or it stays 0 and no
       finite amount of darkening C could make C <= a black A where C is positive)

    FORMULA: Validity darkening (global exposure-down so a transmittance B<=1 exists)
      d = min( 1,  min over pixels (with L(C) > 0) of  L(A) / L(C) )
      C' = C * d       (in [0, 1]; darkening never pushes C above 1)

    FORMULA: Brightness decrease in stops -- how much exposure was removed from C
      stops_down = log2(d)      (<= 0; -1 stop = half the light = ISO 200 -> ISO 100)

    Returns (la, lc_prime, d, stops_down): the (black-floored) luminance of A, the
    darkened luminance of C, the gain d, and the stops decrease (<= 0).
    """
    # luminance of A (no clip needed here; C is what gets scaled, and downward only)
    if scene.ndim == 3:
        la = scene[..., 0] * 0.299 + scene[..., 1] * 0.587 + scene[..., 2] * 0.114
    else:
        la = scene.astype(np.float64)
    lc = normalize_transparency(result)    # luminance of C in [0, 1]

    # floor pure-black (8-bit black) pixels of A to a near-black gray 1/255
    la = la.copy()
    la[la < 0.5 / 255.0] = 1.0 / 255.0

    # largest global gain d <= 1 that makes C' = d*C reach <= A everywhere.
    # only pixels where C > 0 constrain d (C == 0 is already <= any A);
    # divide only there to avoid 0/0 -> nan.
    bright_c = lc > 0.0
    if np.any(bright_c):
        d = min(1.0, float(np.min(la[bright_c] / lc[bright_c])))
    else:
        d = 1.0                            # C is all black: nothing to darken
    lc_prime = lc * d                      # in [0, 1]
    stops_down = float(np.log2(d))         # <= 0
    return la, lc_prime, d, stops_down


# ---------------------------------------------------------------------------
# 2. divide C by A' to recover the blurred transmittance the sensor saw
# ---------------------------------------------------------------------------
def estimate_blurred_transmittance(la, lc_prime):
    """Per-pixel transmittance estimate: T_blur = L(C') / L(A).

    C' was darkened so L(C') <= L(A) everywhere, so the ratio is already a valid
    transmittance in [0, 1]. A is black-floored (never 0), so the division is safe;
    we only clip the final ratio for safety.
    """
    # FORMULA: Transmittance recovery (inverse of Multiply blend) -- get the blurred mask back
    # T_blur = L(C') / L(A)   (L = luma, Rec.601; C' darkened so C' <= A everywhere)
    t_blur = lc_prime / la
    return np.clip(t_blur, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2a. move a small PSF kernel into the frequency domain at the image size
# ---------------------------------------------------------------------------
def _psf2otf(psf, shape):
    """Pad/center a PSF kernel to `shape` and return its FFT (the optical transfer fn).

    FORMULA: PSF -> OTF (Optical Transfer Function) -- H = FFT(psf), used by deconvolve.
    """
    pad = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    # shift so the kernel center sits at pixel (0, 0), as the FFT expects
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)
    return np.fft.fft2(pad)


# ---------------------------------------------------------------------------
# 2b. deconvolve the blur out of T_blur to recover the sharp mask transmittance
# ---------------------------------------------------------------------------
def deconvolve(t_blur, radius, kernel="disk", noise_ratio=0.01):
    """Wiener deconvolution: undo the defocus PSF to get the sharp transmittance.

    radius / kernel must match the forward simulation's PSF.
    noise_ratio is the regularization (noise-to-signal); larger = smoother but
    less sharp, smaller = sharper but noisier/ringy. radius <= 0 means no blur
    was applied, so there is nothing to deconvolve.
    """
    if radius <= 0:
        return np.clip(t_blur, 0.0, 1.0)

    # FORMULA: Wiener deconvolution (Wiener filter / regularized inverse filter)
    # does: remove the defocus blur to recover the SHARP mask transmittance
    #   H = FFT(psf),  G = FFT(T_blur)
    #   F = G * conj(H) / ( |H|^2 + NSR )
    #   T = Re( IFFT(F) )
    psf = build_psf(radius, kernel=kernel)
    otf = _psf2otf(psf, t_blur.shape)                       # H = FFT(psf)

    g = np.fft.fft2(t_blur)                                 # G = FFT(T_blur)
    otf_conj = np.conj(otf)                                 # conj(H)
    f = g * otf_conj / (np.abs(otf) ** 2 + noise_ratio)     # F
    sharp = np.real(np.fft.ifft2(f))                        # T = Re(IFFT(F))
    return np.clip(sharp, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. turn the transmittance map into a black/gray/white mask image
# ---------------------------------------------------------------------------
def transmittance_to_mask(transmittance):
    """Map transmittance [0, 1] to a grayscale mask float array (white = transparent)."""
    # transmittance already encodes white=1 / black=0, so this is a passthrough clip;
    # kept as its own step for symmetry with normalize_transparency in the forward file.
    return np.clip(transmittance, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. write a float array back out as a grayscale jpeg file
# ---------------------------------------------------------------------------
def write_image(arr, path, quality=95):
    """Write a float array in [0, 1] to a grayscale jpeg file."""
    out = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(out, mode="L").save(path, "JPEG", quality=quality)


# ---------------------------------------------------------------------------
# pipeline glue
# ---------------------------------------------------------------------------
def recover_mask(scene_path, result_path, mask_out_path,
                 mask_h, mask_w, blur_radius, kernel="disk", noise_ratio=0.01):
    """Recover mask B from scene A and result C, save it at (mask_h, mask_w).

    Returns (mask, d, stops_down): the recovered mask, the global darkening gain
    applied to C to make a valid B exist, and that decrease in exposure stops (<= 0).
    """
    # 1. read both jpeg inputs (A and C share the same resolution)
    scene = read_image(scene_path, grayscale=False)     # image A, shape (a1, a2, 3)
    result = read_image(result_path, grayscale=False)   # image C, shape (a1, a2, 3)

    # 1b. darken C if A is darker than C anywhere, so a valid B (B <= 1) exists.
    #     lc_prime is C's luminance after darkening (in [0, 1]); la is black-floored.
    la, lc_prime, d, stops_down = darken_C_to_valid(scene, result)

    # 2. divide C' / A to get the blurred transmittance the sensor recorded
    t_blur = estimate_blurred_transmittance(la, lc_prime)

    # 3. deconvolve the defocus PSF to get the sharp mask transmittance
    transmittance = deconvolve(t_blur, blur_radius, kernel=kernel, noise_ratio=noise_ratio)

    # 4. map to a black/gray/white mask (still at A's resolution)
    mask = transmittance_to_mask(transmittance)

    # 5. FORMULA: Recovered mask -- B = clip(T,0,1) resized (a1 x a2 -> b1 x b2)
    mask = resize_image(mask, mask_h, mask_w)

    # 6. write the recovered mask out as a grayscale jpeg
    write_image(mask, mask_out_path)

    return mask, d, stops_down


# ---------------------------------------------------------------------------
# all tunable inputs live here
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- file paths ---
    scene_path = "/Users/liuhaoyu/Documents/毕业旅行/UGSL4192.jpg"        # the plain photo of the scene (image A)
    result_path = "/Users/liuhaoyu/Downloads/project/lens/orgmask.JPG"       # the photo taken through the mask (image C)
    mask_out_path = "/Users/liuhaoyu/Downloads/project/lens/genmask.JPG"   # where to save the recovered mask (image B)

    # --- mask output resolution (b1 x b2) ---
    # C is assumed to be the same resolution as A; only B has its own size.
    mask_h = 480      # b1
    mask_w = 640      # b2

    # --- KNOB A: PSF shape (must match the forward simulation) ---
    kernel = "disk"   # "disk" (lens blur) or "gauss"

    # --- KNOB B: where does the blur radius come from? (must match forward) ---
    radius_mode = "manual"    # "manual" or "physical"

    manual_blur_radius = 8.0  # used when radius_mode == "manual"

    # used when radius_mode == "physical" (all distances in mm)
    aperture_diameter_mm = 5.0
    focal_length_mm = 50.0
    pixel_pitch_mm = 0.0043
    mask_to_lens_mm = 30.0
    focus_distance_mm = 2000.0

    if radius_mode == "physical":
        blur_radius = physical_blur_radius(
            aperture_diameter_mm, focal_length_mm, pixel_pitch_mm,
            mask_to_lens_mm, focus_distance_mm,
        )
        print(f"physical blur radius = {blur_radius:.2f} px")
    else:
        blur_radius = manual_blur_radius

    # --- deconvolution regularization ---
    # larger -> smoother / safer, smaller -> sharper but more ringing & noise
    noise_ratio = 0.01

    mask, d, stops_down = recover_mask(
        scene_path, result_path, mask_out_path,
        mask_h, mask_w, blur_radius, kernel=kernel, noise_ratio=noise_ratio)
    print(f"Done. Saved recovered mask to {mask_out_path}")
    if d < 1.0:
        print(f"A was darker than C somewhere: darkened C by {d:.3f}x "
              f"({stops_down:.2f} stops) so a valid mask B exists")
    else:
        print("A was already >= C everywhere; no darkening needed (0 stops)")
