import cv2
import numpy as np
import os
import argparse

#Befor Run code
#pip install opencv-python-headless numpy
#python anti-aliasing.py --input input.jpg --outdir results --ssaa_factor 3

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # convert to RGB for nicer saving/visualizing later (cv2 uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(path, img):
    # expects RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def resize_nearest(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def resize_bilinear(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def resize_bicubic(img, new_w, new_h):
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def supersample_ssaa(img, target_w, target_h, factor=2, prefilter='gaussian'):
    """Supersampling: upscale by factor, optionally blur, then downsample to target."""
    # upscale
    up_w, up_h = target_w * factor, target_h * factor
    up = cv2.resize(img, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    # prefilter
    if prefilter == 'gaussian':
        k = max(3, int(2*factor+1))
        up = cv2.GaussianBlur(up, (k, k), sigmaX=0)
    elif prefilter == 'box':
        k = factor
        up = cv2.blur(up, (k, k))
    # downsample with linear
    res = cv2.resize(up, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return res

def prefilter_downsample(img, target_w, target_h, kernel='gaussian', sigma=1.0):
    """Apply low-pass then downsample directly."""
    if kernel == 'gaussian':
        k = int(max(3, (sigma*3)//1*2+1))
        imgf = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    else:
        imgf = cv2.blur(img, (3,3))
    return cv2.resize(imgf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def fxaa_like(img, strength=1.0, blur_ksize=7):
    """
    Approximate FXAA: detect edges, blur edges, blend.
    Not a full FXAA, but gives edge smoothing without blurring whole image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel magnitude
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    # Normalize and threshold to get mask
    mag_norm = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # adaptive threshold: higher strength -> more aggressive smoothing
    thresh = 0.15 * strength
    mask = (mag_norm > thresh).astype(np.float32)
    # blur whole image (but we'll composite only on edges)
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), sigmaX=0)
    # composite: where mask=1 use blurred, else original
    mask3 = np.stack([mask]*3, axis=2)
    out = (img * (1.0 - mask3) + blurred * mask3).astype(np.uint8)
    return out

def downscale_compare_methods(img, target_w, target_h, ssaa_factor=3, fxaa_strength=1.0):
    results = {}
    results['nearest'] = resize_nearest(img, target_w, target_h)
    results['bilinear'] = resize_bilinear(img, target_w, target_h)
    results['bicubic'] = resize_bicubic(img, target_w, target_h)
    results['ssaa_gaussian'] = supersample_ssaa(img, target_w, target_h, factor=ssaa_factor, prefilter='gaussian')
    results['ssaa_box'] = supersample_ssaa(img, target_w, target_h, factor=ssaa_factor, prefilter='box')
    results['prefilter_gaussian'] = prefilter_downsample(img, target_w, target_h, kernel='gaussian', sigma=1.5)
    results['fxaa_like'] = fxaa_like(resize_bilinear(img, target_w, target_h), strength=fxaa_strength, blur_ksize=5)
    return results

def make_comparison_strip(result_dict):
    """Create a horizontal strip with labels (simple)."""
    imgs = []
    for k in result_dict:
        imgs.append(result_dict[k])
    # pad to same height/width
    heights = [im.shape[0] for im in imgs]
    max_h = max(heights)
    padded = []
    for im in imgs:
        h, w = im.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h-h, w, 3), dtype=np.uint8) + 255
            im2 = np.vstack([im, pad])
        else:
            im2 = im
        padded.append(im2)
    strip = np.hstack(padded)
    return strip

def main():
    parser = argparse.ArgumentParser(description="Simulate anti-aliasing methods on raster images.")
    parser.add_argument('--input', '-i', required=True, help='Path to input image')
    parser.add_argument('--outdir', '-o', default='aa_results', help='Output directory')
    parser.add_argument('--target_w', type=int, default=400, help='Target width (px)')
    parser.add_argument('--target_h', type=int, default=300, help='Target height (px)')
    parser.add_argument('--ssaa_factor', type=int, default=3, help='SSAA upsample factor')
    parser.add_argument('--fxaa_strength', type=float, default=1.0, help='FXAA-like strength')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = read_image(args.input)
    print(f"Input image shape: {img.shape}")

    res = downscale_compare_methods(img, args.target_w, args.target_h, ssaa_factor=args.ssaa_factor, fxaa_strength=args.fxaa_strength)

    # save each
    for name, im in res.items():
        outpath = os.path.join(args.outdir, f"{name}.png")
        save_image(outpath, im)
        print(f"Saved {outpath}")

    # also save a comparison strip
    strip = make_comparison_strip(res)
    save_image(os.path.join(args.outdir, "comparison_strip.png"), strip)
    print(f"Saved comparison_strip.png in {args.outdir}")

if __name__ == '__main__':
    main()
