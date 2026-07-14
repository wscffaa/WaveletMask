import cv2
import random
import torch


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Convert numpy image(s) to tensor."""

    def _to_tensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            tensor = tensor.float()
        return tensor

    if isinstance(imgs, list):
        return [_to_tensor(img, bgr2rgb, float32) for img in imgs]
    return _to_tensor(imgs, bgr2rgb, float32)


def paired_random_crop(img_gts, img_lqs, gt_patch_size_w, gt_patch_size_h, scale, gt_path):
    """Paired random crop keeping GT/LQ alignment."""
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape

    if h_lq == 0 or w_lq == 0:
        raise ValueError('LQ image has invalid spatial size.')

    inferred_scale_h = max(1, round(h_gt / h_lq))
    inferred_scale_w = max(1, round(w_gt / w_lq))
    inferred_scale = inferred_scale_h if inferred_scale_h == inferred_scale_w else scale
    effective_scale = max(1, inferred_scale)

    lq_patch_size_w = max(1, gt_patch_size_w // effective_scale)
    lq_patch_size_h = max(1, gt_patch_size_h // effective_scale)

    top = random.randint(0, h_lq - lq_patch_size_h)
    left = random.randint(0, w_lq - lq_patch_size_w)

    img_lqs = [
        v[top:top + lq_patch_size_h, left:left + lq_patch_size_w, ...]
        for v in img_lqs
    ]
    top_gt, left_gt = int(top * effective_scale), int(left * effective_scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size_h, left_gt:left_gt + gt_patch_size_w, ...]
        for v in img_gts
    ]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True):
    """Apply random flip/rotation augmentations."""
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    return [_augment(img) for img in imgs]
