"""Simulate photographing a scene through an out-of-focus mask placed in front of the lens.

Physical model
--------------
Image A : the scene imaged sharply onto the CMOS (camera focuses on the scene).
Image B : a black/gray/white mask placed in front of the lens, at some distance
          from the sensor, so it is OUT of focus.
Image C : the photo taken with the mask in place.

Because the mask is not on the focal plane, its shadow on the sensor is blurred
by the defocus point-spread-function (PSF). The PSF shape is the aperture shape
(a disk for a round aperture); its radius grows with the mask-to-sensor distance
and the aperture diameter.

So instead of a plain Photoshop "multiply" (C = A * B), we do:

    C = A * blur(B)

where blur() is a convolution with the defocus PSF.

Math formulas used in this file  (grep "FORMULA:" to jump between them)
-----------------------------------------------------------------------
(i,j) = pixel index, * = pixel-wise product, (.) ⊛ (.) = 2-D convolution.

  FORMULA: Luma / luminance  (a.k.a. Rec.601 / ITU-R BT.601 luma)
    does: collapse an RGB pixel to one brightness value
    L = 0.299*R + 0.587*G + 0.114*B                         [in normalize_transparency]

  FORMULA: Transmittance map
    does: read the mask as "how much light passes" (white=1 pass, black=0 block)
    T(i,j) = L(i,j)                                         [in normalize_transparency]

  FORMULA: Disk / pillbox PSF  (a.k.a. circle-of-confusion / bokeh kernel)
    does: shape of an out-of-focus spot for a round aperture
    psf(x,y) = 1 if x^2 + y^2 <= r^2 else 0,  then psf /= sum(psf)   [in build_psf]

  FORMULA: Gaussian PSF  (a.k.a. Gaussian blur kernel)
    does: smooth approximation of the out-of-focus spot
    psf(x,y) = exp(-(x^2 + y^2)/(2*sigma^2)),  sigma = r/3,  psf /= sum(psf)  [in build_psf]

  FORMULA: 2-D convolution  (via FFT, by the convolution theorem)
    does: apply the defocus blur to the mask
    T_blur = T ⊛ psf                                        [in apply_psf, scipy.fftconvolve]

  FORMULA: Multiply blend  (Photoshop "Multiply"; physically Beer-Lambert transmittance)
    does: darken the scene by the blurred mask -> final image C
    C = A * T_blur          (T_blur broadcast over R,G,B)   [in multiply]

  FORMULA: Circle of Confusion (CoC) for a near mask  (geometric / thin-lens defocus)
    does: convert real optics into a blur radius in pixels
    diameter_px = D*f*(s - t) / (p*t*s),   r = diameter_px / 2   [in physical_blur_radius]
    D=aperture mm, f=focal mm, p=pixel pitch mm/px, t=mask-to-lens mm, s=focus distance mm

  FORMULA: Black floor (avoid 0*gain=0)
    does: lift pure-black (0,0,0) pixels of C to a near-black gray (1,1,1)=1/255
    C[pixel < 0.5/255] = 1/255                             [in match_brightness]

  FORMULA: Average luminance / mean brightness
    does: one brightness number for a whole image
    mean = mean( L )                                       [in match_brightness]

  FORMULA: Exposure gain (linear exposure scaling)
    does: brighten C so its average brightness equals A's (what raising ISO does)
    gain = mean_A / mean_C,   C_matched = clip(C * gain)   [in match_brightness]

  FORMULA: Exposure value (EV) / stops
    does: how many exposure levels to add (1 stop = 2x light = ISO100->ISO200)
    stops = log2(gain)                                     [in match_brightness]
"""

import numpy as np
from scipy.signal import fftconvolve
from PIL import Image


# ---------------------------------------------------------------------------
# 1. read a jpeg file into a float array
# ---------------------------------------------------------------------------
def read_image(path, grayscale=False):
    """Read a jpeg file and return it as a float array in range [0, 1].

    Returns an array of shape (h, w, 3) for color, or (h, w) for grayscale.
    """
    img = Image.open(path)
    img = img.convert("L" if grayscale else "RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


# ---------------------------------------------------------------------------
# 2. resize an image to a target pixel resolution
# ---------------------------------------------------------------------------
def resize_image(arr, target_h, target_w):
    """Resize a float array to (target_h, target_w).

    Used to map the mask B (b1 x b2) onto the scene's field of view (a1 x a2).
    """
    # remember whether the input is grayscale so we can restore the shape
    is_gray = (arr.ndim == 2)

    pil_mode = "L" if is_gray else "RGB"
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode=pil_mode)
    # PIL size is (width, height)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float64) / 255.0
    return out


# ---------------------------------------------------------------------------
# 3. normalize the mask into a transmittance map (white = transparent = 1)
# ---------------------------------------------------------------------------
def normalize_transparency(mask):
    """Turn the mask into a transmittance map in [0, 1].

    white (255) -> 1.0 (fully transparent / passes all light)
    black (0)   -> 0.0 (fully opaque / blocks all light)
    gray        -> partial transmittance

    If the mask is color it is collapsed to a single luminance channel, because
    a physical neutral mask transmits the same fraction of every wavelength.
    """
    if mask.ndim == 3:
        # FORMULA: Luma (Rec.601) -- collapse RGB to one brightness value
        # L = 0.299*R + 0.587*G + 0.114*B
        mask = mask[..., 0] * 0.299 + mask[..., 1] * 0.587 + mask[..., 2] * 0.114
    transmittance = np.clip(mask, 0.0, 1.0)
    return transmittance


# ---------------------------------------------------------------------------
# 4a. build the defocus PSF (point-spread-function) kernel
# ---------------------------------------------------------------------------
def build_psf(radius, kernel="disk"):
    """Build a normalized defocus PSF kernel.

    KNOB A -- the PSF *shape*:
        "disk"  -> flat circular kernel, the shape of a real round aperture
                   (this is the "lens blur" / bokeh look)
        "gauss" -> gaussian kernel, a smoother and cheaper approximation

    radius is the blur radius in pixels. The kernel is normalized so a uniform
    mask passes through unchanged (energy preserving).
    """
    if kernel == "disk":
        # FORMULA: Disk/pillbox PSF (circle-of-confusion / bokeh) -- round-aperture blur spot
        # psf(x,y) = 1 if x^2 + y^2 <= r^2 else 0
        r = int(np.ceil(radius))
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        psf = ((x * x + y * y) <= radius * radius).astype(np.float64)
    elif kernel == "gauss":
        # FORMULA: Gaussian PSF (Gaussian blur) -- smooth approximation of the blur spot
        # psf(x,y) = exp(-(x^2 + y^2) / (2*sigma^2)),  sigma = r/3
        # (radius treated as ~3 sigma so the kernel covers the same footprint)
        sigma = radius / 3.0
        r = int(np.ceil(radius))
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        psf = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    else:
        raise ValueError("kernel must be 'disk' or 'gauss'")

    psf /= psf.sum()  # normalize: sum(psf) = 1, so a uniform mask is unchanged
    return psf


# ---------------------------------------------------------------------------
# 4b. convolve the transmittance map with the defocus PSF
# ---------------------------------------------------------------------------
def apply_psf(transmittance, radius, kernel="disk"):
    """Blur the transmittance map with the defocus PSF.

    radius <= 0 means the mask is in perfect focus -> no blur (plain multiply).
    Returns the blurred transmittance map, same shape, still in [0, 1].
    """
    if radius <= 0:
        return transmittance

    # FORMULA: 2-D convolution (convolution theorem, done via FFT) -- applies the blur
    # T_blur = T ⊛ psf
    psf = build_psf(radius, kernel=kernel)
    blurred = fftconvolve(transmittance, psf, mode="same")
    return np.clip(blurred, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4c. KNOB B -- compute the blur radius from real physical optics
# ---------------------------------------------------------------------------
def physical_blur_radius(aperture_diameter_mm, focal_length_mm, pixel_pitch_mm,
                         mask_to_lens_mm, focus_distance_mm):
    """Circle-of-confusion radius (in pixels) for a mask placed in front of the lens.

    FORMULA: Circle of Confusion (CoC) -- geometric / thin-lens defocus model.
    Converts real optics into a pixel blur radius for build_psf().

    Geometry: the camera is focused on the scene at distance `focus_distance_mm`,
    so the scene is sharp. The mask sits `mask_to_lens_mm` in front of the lens,
    far off the focal plane, so its shadow on the sensor is blurred. Tracing the
    aperture cone back to the mask plane and projecting onto the sensor gives:

        blur_diameter_pixels = D * f * (s - t) / (p * t * s)

        D = aperture_diameter_mm   (bigger aperture  -> more blur)
        f = focal_length_mm
        p = pixel_pitch_mm         (sensor mm per pixel)
        t = mask_to_lens_mm        (mask closer to lens -> more blur)
        s = focus_distance_mm      (scene / focus distance; s -> inf simplifies to D*f/(p*t))

    Returns the radius in pixels (= diameter / 2), clamped at >= 0.
    """
    D = aperture_diameter_mm
    f = focal_length_mm
    p = pixel_pitch_mm
    t = mask_to_lens_mm
    s = focus_distance_mm

    diameter_px = D * f * (s - t) / (p * t * s)
    radius_px = diameter_px / 2.0
    return max(radius_px, 0.0)


# ---------------------------------------------------------------------------
# 5. multiply the scene by the (blurred) mask transmittance
# ---------------------------------------------------------------------------
def multiply(scene, transmittance):
    """Combine scene A with the blurred mask transmittance: C = A * B_blur.

    Broadcasts the single-channel transmittance across the scene's color channels.
    """
    # FORMULA: Multiply blend (Photoshop "Multiply"; Beer-Lambert transmittance) -- final C
    # C(i,j) = A(i,j) * T_blur(i,j)   (T_blur broadcast over R,G,B)
    if scene.ndim == 3:
        transmittance = transmittance[..., np.newaxis]  # (h, w, 1) -> broadcast
    result = scene * transmittance
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5b. match the average brightness of C back to A, and report the exposure cost
# ---------------------------------------------------------------------------
def _mean_brightness_C(result, gain):
    """Average brightness of C after applying `gain` and clipping to [0, 1]."""
    return float(normalize_transparency(np.clip(result * gain, 0.0, 1.0)).mean())


def _floor_black(result):
    """Lift pure-black (0,0,0) pixels of C to a near-black gray (1,1,1) = 1/255.

    "(0,0,0)" means black as an 8-bit image value (< 0.5/255), which also absorbs the
    tiny floating-point dust FFT convolution leaves in fully-masked areas. The 1/255
    floor avoids the "0 * gain = 0" dead pixel: a fully-blocked region then behaves
    like a sensor noise floor that brightens to a gray as exposure rises, instead of
    being frozen at pure black. Returns a copy; the original array is untouched.
    """
    black_level = 0.5 / 255.0
    gray = 1.0 / 255.0
    out = result.copy()
    if result.ndim == 3:
        black_mask = np.all(result < black_level, axis=-1)
    else:
        black_mask = (result < black_level)
    out[black_mask] = gray
    return out


def match_brightness(scene, result, mode="linear", tol=1e-4, max_iter=60):
    """Scale result C so its average brightness equals scene A's.

    The mask blocks light, so C is darker than A. Raising the camera's exposure
    (e.g. ISO 100 -> ISO 200) multiplies every pixel by a constant gain. We report
    the matched image, the gain, and how many exposure "stops" that costs.

    Black-floor rule: every pure-black (0,0,0) pixel of the old C is first lifted to a
    near-black gray (1,1,1) = 1/255 (a value on the 0-255 scale, NOT white). This
    avoids "0 * gain = 0" freezing blocked regions at pure black; and since no pixel
    is exactly 0, a large enough gain can push the whole image up to white, so the
    achievable average covers (0, 1] and A's brightness is always reachable with a
    finite gain. No gain cap is needed.

    mode:
      "linear" -- true exposure compensation: gain = mean_A / mean_C, then clip.
      "exact"  -- bisection: over-expose until C2's measured average equals A's
                  (highlights clip to white on purpose). Always converges.

    FORMULA: Black floor (avoid 0*gain=0) -- lift blocked pixels off pure black
      C[pixel < 0.5/255] = 1/255          (8-bit black -> near-black gray (1,1,1))

    FORMULA: Average luminance / mean brightness -- one brightness value per image
      mean_A = mean( L(A) ),   mean_C = mean( L(C) )      L = luma (Rec.601)

    FORMULA: Exposure gain (linear exposure scaling) -- factor that equalizes brightness
      gain = mean_A / mean_C                              [mode="linear"]
      C_matched = clip( C * gain, 0, 1 )

    FORMULA: Exact brightness match (root-finding / bisection on the measured mean)
      solve gain such that  mean( L( clip(C*gain) ) ) = mean_A   [mode="exact"]
      (monotonic from 0 to 1 in gain, so bisection always converges)

    FORMULA: Exposure value (EV) / stops -- how much exposure to add
      stops = log2(gain)
      (1 stop = 2x the light = ISO 100 -> ISO 200; positive stops = brighten)

    Returns (C_matched, gain, stops).
    """
    mean_a = float(normalize_transparency(scene).mean())   # average brightness of A

    # avoid "0 * gain = 0": lift pure-black (0,0,0) pixels of the old C to 1/255 gray
    result = _floor_black(result)

    mean_c = _mean_brightness_C(result, 1.0)               # average brightness of C
    if mean_c <= 0.0:
        return np.clip(result, 0.0, 1.0), 1.0, 0.0

    gain_linear = mean_a / mean_c

    if mode == "linear":
        gain = gain_linear
    elif mode == "exact":
        # measured brightness(gain) is monotonic from 0 (gain->0) to 1.0 (gain->inf),
        # so any mean_A in (0, 1) is reachable with a finite gain -- no cap needed.
        lo, hi = 0.0, max(gain_linear, 1.0)
        while _mean_brightness_C(result, hi) < mean_a:
            hi *= 2.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if _mean_brightness_C(result, mid) < mean_a:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, hi):
                break
        gain = 0.5 * (lo + hi)
    else:
        raise ValueError("mode must be 'linear' or 'exact'")

    matched = np.clip(result * gain, 0.0, 1.0)
    stops = np.log2(gain)
    return matched, gain, stops


# ---------------------------------------------------------------------------
# 6. write a float array back out as a jpeg file
# ---------------------------------------------------------------------------
def write_image(arr, path, quality=95):
    """Write a float array in [0, 1] to a jpeg file."""
    out = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    mode = "L" if out.ndim == 2 else "RGB"
    Image.fromarray(out, mode=mode).save(path, "JPEG", quality=quality)


# ---------------------------------------------------------------------------
# pipeline glue
# ---------------------------------------------------------------------------
def simulate(scene_path, mask_path, out_path, blur_radius, kernel="disk",
             out_matched_path=None, brightness_mode="linear"):
    """Run the full A + B -> C simulation and save the result.

    Saves the raw (darkened) C to out_path. If out_matched_path is given, also
    saves a brightness-matched C2 (see match_brightness for brightness_mode) and
    returns the exposure cost in stops. Returns (result, matched, gain, stops).
    """
    # 1. read both jpeg inputs
    scene = read_image(scene_path, grayscale=False)   # image A, shape (a1, a2, 3)
    mask = read_image(mask_path, grayscale=False)     # image B, shape (b1, b2, 3)

    a1, a2 = scene.shape[0], scene.shape[1]

    # 2. resize the mask onto the scene's field of view (b1 x b2 -> a1 x a2)
    mask = resize_image(mask, a1, a2)

    # 3. normalize the mask into a transmittance map (white = transparent)
    transmittance = normalize_transparency(mask)

    # 4. blur the mask with the defocus PSF (the distance effect)
    transmittance = apply_psf(transmittance, blur_radius, kernel=kernel)

    # 5. multiply the scene by the blurred transmittance
    result = multiply(scene, transmittance)           # image C, shape (a1, a2, 3)

    # 6. write the raw (darkened) result out as a jpeg
    write_image(result, out_path)

    # 5b. also build a brightness-matched C2 and the exposure cost in stops
    matched, gain, stops = match_brightness(scene, result, mode=brightness_mode)
    if out_matched_path is not None:
        write_image(matched, out_matched_path)

    return result, matched, gain, stops


# ---------------------------------------------------------------------------
# all tunable inputs live here
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- file paths ---
    scene_path = "/Users/liuhaoyu/Documents/毕业旅行/UGSL4192.jpg"          # the plain photo of the scene (image A)
    mask_path = "/Users/liuhaoyu/Downloads/project/lens/genmask.JPG"           # the black/gray/white mask (image B)
    out_path = "/Users/liuhaoyu/Downloads/project/lens/regenmask.JPG"            # raw darkened result (image C)
    out_matched_path = "/Users/liuhaoyu/Downloads/project/lens/regenmask.JPG"  # C re-exposed to match A's brightness

    # --- brightness matching mode for the C2 output ---
    # (pure-black (0,0,0) pixels of C are lifted to a near-black gray 1/255 first,
    #  to avoid "0 * gain = 0" freezing blocked regions at pure black)
    #   "linear" -> true exposure compensation (gain = mean_A/mean_C)
    #   "exact"  -> force C2's measured average to equal A's by over-exposing/
    #               clipping on purpose (root-finding); always reachable, no cap
    brightness_mode = "linear"

    # --- KNOB A: PSF shape ---
    #   "disk"  -> flat circular kernel, closest to a real round aperture (lens blur)
    #   "gauss" -> gaussian kernel, smoother and faster
    kernel = "disk"

    # --- KNOB B: where does the blur radius come from? ---
    #   "manual"   -> use manual_blur_radius below (you tune it by hand)
    #   "physical" -> compute it from the real optical parameters below
    radius_mode = "manual"

    # used when radius_mode == "manual": circle-of-confusion radius in PIXELS
    #   0           -> mask in perfect focus, behaves like a plain Photoshop multiply
    #   small (2-5) -> mask close to the focal plane, soft edges
    #   large (15+) -> mask far from the sensor / wide aperture, very blurred shadow
    manual_blur_radius = 8.0

    # used when radius_mode == "physical": real optics (all distances in mm)
    aperture_diameter_mm = 5.0      # D: lens aperture diameter
    focal_length_mm = 50.0          # f: lens focal length
    pixel_pitch_mm = 0.0043         # p: sensor size per pixel (e.g. ~4.3 um)
    mask_to_lens_mm = 30.0          # t: distance from mask to lens
    focus_distance_mm = 2000.0      # s: distance the camera is focused at (the scene)

    if radius_mode == "physical":
        blur_radius = physical_blur_radius(
            aperture_diameter_mm, focal_length_mm, pixel_pitch_mm,
            mask_to_lens_mm, focus_distance_mm,
        )
        print(f"physical blur radius = {blur_radius:.2f} px")
    else:
        blur_radius = manual_blur_radius

    # the output resolution always equals the scene's resolution (a1 x a2),
    # this is handled automatically inside simulate().
    result, matched, gain, stops = simulate(
        scene_path, mask_path, out_path, blur_radius, kernel=kernel,
        out_matched_path=out_matched_path, brightness_mode=brightness_mode,
    )
    print(f"Done. Saved raw result to {out_path}")
    print(f"Saved brightness-matched result to {out_matched_path}")
    print(f"Exposure gain = {gain:.3f}x  ->  raise exposure by {stops:.2f} stops "
          f"(ISO levels) to match A's brightness")
