"""
数据增强脚本：将 test_images/ 中的锥桶图片做颜色变换与几何增强，
输出到 output_images/。

目标：10 张原图 × 2^N 种变换组合 = 10 × 2^N 张输出图片
  N = 增强变换种类数（如水平翻转、Affine、尺度缩放等）
"""

from itertools import product
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR = Path(__file__).parent / "output_images"


# 定义 N 种增强（每种增强都有开/关两个状态）
AUG_TYPES = ["flip_h", "affine", "scale", "brightness"]


def red_to_green(image: np.ndarray) -> np.ndarray:
    """
    将红/橘色区域偏移到绿色区域（HSV hue shift）。

    思路：
    1) 转 HSV
    2) 针对红/橘色区域建立 mask
    3) 将该区域 hue 平移到绿色附近，并提高饱和度
    4) 转回 BGR
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色在 HSV hue 上通常分布在两端：低值段和高值段
    lower_red1 = np.array([0, 50, 30], dtype=np.uint8)
    upper_red1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 50, 30], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 将红/橘色 hue 映射到绿色（约 60）
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)

    target_green_hue = 60
    red_pixels = red_mask > 0
    hue[red_pixels] = target_green_hue
    sat[red_pixels] = np.clip(sat[red_pixels] + 20, 0, 255)

    hsv[:, :, 0] = hue.astype(np.uint8)
    hsv[:, :, 1] = sat.astype(np.uint8)

    green_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return green_img


def apply_augmentations(image: np.ndarray, aug_flags: dict) -> np.ndarray:
    """
    根据 aug_flags 对 image 应用几何/光照增强。
    aug_flags: 如 {"flip_h": True, "affine": False, "scale": True, "brightness": False}
    """
    out = image.copy()
    h, w = out.shape[:2]

    if aug_flags.get("flip_h", False):
        out = cv2.flip(out, 1)

    if aug_flags.get("affine", False):
        # 轻量仿射：平移 + 小角度旋转
        center = (w / 2, h / 2)
        m_rot = cv2.getRotationMatrix2D(center, angle=8, scale=1.0)
        m_rot[0, 2] += 0.03 * w
        m_rot[1, 2] += 0.02 * h
        out = cv2.warpAffine(
            out,
            m_rot,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    if aug_flags.get("scale", False):
        # 缩放后再裁剪/填充回原尺寸
        scale_factor = 1.12
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 中心裁剪回原图尺寸
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        out = resized[start_y:start_y + h, start_x:start_x + w]

    if aug_flags.get("brightness", False):
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.int16)
        v = np.clip(v + 25, 0, 255)
        hsv[:, :, 2] = v.astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return out


def generate_aug_combinations(aug_types: list[str]):
    """生成 2^N 种增强组合（每种变换开/关）。"""
    combinations = []
    for bits in product([False, True], repeat=len(aug_types)):
        flags = dict(zip(aug_types, bits))
        combinations.append(flags)
    return combinations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(INPUT_DIR.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    if len(images) != 10:
        print(f"Warning: expected 10 images in test_images/, found {len(images)}")

    aug_combos = generate_aug_combinations(AUG_TYPES)
    expected_per_image = 2 ** len(AUG_TYPES)
    print(f"Using augment types: {AUG_TYPES}")
    print(f"N={len(AUG_TYPES)}, combinations per image=2^N={expected_per_image}")

    total_written = 0
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable image: {img_path.name}")
            continue

        # 先进行红->绿颜色变换，再做组合增强
        green_img = red_to_green(img)

        stem = img_path.stem
        ext = ".jpg"

        for idx, flags in enumerate(aug_combos):
            aug_img = apply_augmentations(green_img, flags)

            # 组合编码便于追溯
            bit_code = "".join("1" if flags[k] else "0" for k in AUG_TYPES)
            out_name = f"{stem}_green_aug{idx:02d}_{bit_code}{ext}"
            out_path = OUTPUT_DIR / out_name

            ok = cv2.imwrite(str(out_path), aug_img)
            if ok:
                total_written += 1

    expected_total = len(images) * expected_per_image
    print(f"Done. Wrote {total_written} images to: {OUTPUT_DIR}")
    print(f"Expected (len(images) * 2^N): {expected_total}")


if __name__ == "__main__":
    main()
