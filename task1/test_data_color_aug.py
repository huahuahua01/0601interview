from pathlib import Path

import cv2

import data_color_aug as aug


def _collect_images(directory: Path):
    return sorted(
        [p for p in directory.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )


def _expected_output_names(input_images, aug_types):
    combos = aug.generate_aug_combinations(aug_types)
    names = []
    for img_path in input_images:
        stem = img_path.stem
        for idx, flags in enumerate(combos):
            bit_code = "".join("1" if flags[k] else "0" for k in aug_types)
            names.append(f"{stem}_green_aug{idx:02d}_{bit_code}.jpg")
    return names


def run_tests():
    input_images = _collect_images(aug.INPUT_DIR)
    output_images = _collect_images(aug.OUTPUT_DIR)

    expected_per_image = 2 ** len(aug.AUG_TYPES)
    expected_total = len(input_images) * expected_per_image

    print("=== data_color_aug tests ===")
    print(f"Input images: {len(input_images)}")
    print(f"Output images: {len(output_images)}")
    print(f"AUG_TYPES: {aug.AUG_TYPES}")
    print(f"Expected per image: {expected_per_image}")
    print(f"Expected total: {expected_total}")

    all_pass = True

    # Test 1: 输入目录不为空
    t1 = len(input_images) > 0
    print(f"[{'PASS' if t1 else 'FAIL'}] test_input_not_empty")
    all_pass &= t1

    # Test 2: 输出总数量正确
    t2 = len(output_images) == expected_total
    print(f"[{'PASS' if t2 else 'FAIL'}] test_output_total_count")
    all_pass &= t2

    # Test 3: 每个原图都有 2^N 张输出
    t3 = True
    for img_path in input_images:
        prefix = f"{img_path.stem}_green_aug"
        cnt = sum(1 for p in output_images if p.name.startswith(prefix))
        if cnt != expected_per_image:
            t3 = False
            print(f"  -> {img_path.name}: got {cnt}, expected {expected_per_image}")
    print(f"[{'PASS' if t3 else 'FAIL'}] test_each_input_has_full_combinations")
    all_pass &= t3

    # Test 4: 文件名是否完整匹配预期集合
    expected_names = set(_expected_output_names(input_images, aug.AUG_TYPES))
    actual_names = set(p.name for p in output_images)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    t4 = (len(missing) == 0) and (len(extra) == 0)
    print(f"[{'PASS' if t4 else 'FAIL'}] test_output_name_set")
    if not t4:
        if missing:
            print(f"  missing sample: {missing[:5]}")
        if extra:
            print(f"  extra sample: {extra[:5]}")
    all_pass &= t4

    # Test 5: 抽样检查图片可读、尺寸有效
    t5 = True
    for p in output_images[: min(20, len(output_images))]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            t5 = False
            print(f"  unreadable: {p.name}")
            break
    print(f"[{'PASS' if t5 else 'FAIL'}] test_output_images_readable")
    all_pass &= t5

    print("============================")
    print("ALL PASS" if all_pass else "SOME TESTS FAILED")

    return all_pass


if __name__ == "__main__":
    run_tests()
