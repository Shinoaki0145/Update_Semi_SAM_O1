import math

import torch
import torch.nn.functional as F


def validate_ucp_symgd_config(backbone, start_round, scale_min, scale_max,
                              confidence, weight, enabled=True):
    if start_round < 1:
        raise ValueError("ucp_start_round must be at least 1")
    if not 0 < scale_min <= scale_max <= 1:
        raise ValueError("UCP scales must satisfy 0 < min <= max <= 1")
    if not 0 <= confidence <= 1:
        raise ValueError("symgd_confidence must be between 0 and 1")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("symgd_weight must be finite and non-negative")
    if enabled and backbone not in {"mt", "uamt"}:
        raise ValueError("--ucp_symgd supports only mt and uamt")


def ucp_symgd_enabled(enabled, round_num, start_round):
    return enabled and round_num >= start_round


def make_center_masks(images, scale_min, scale_max, generator=None):
    if images.ndim != 5 or images.shape[0] == 0:
        raise ValueError("images must have shape [N, C, D, H, W] with N > 0")
    validate_ucp_symgd_config("mt", 1, scale_min, scale_max, 0, 0)
    count = images.shape[0]
    spatial = images.shape[-3:]
    if scale_min == scale_max:
        ratios = images.new_full((count, 3), scale_min)
    else:
        ratios = torch.empty((count, 3), device=images.device).uniform_(
            scale_min, scale_max, generator=generator)
    lengths = (ratios * images.new_tensor(spatial)).floor().long().clamp(min=1)
    mask = images.new_zeros((count, 1, *spatial))
    for index, size in enumerate(lengths.tolist()):
        starts = [(extent - side) // 2 for extent, side in zip(spatial, size)]
        d, h, w = starts
        sd, sh, sw = size
        mask[index, :, d:d + sd, h:h + sh, w:w + sw] = 1
    return mask


def unified_copy_paste(labeled_images, labeled_labels, unlabeled_images,
                       unlabeled_labels, scale_min, scale_max, generator=None):
    if labeled_images.ndim != 5 or unlabeled_images.ndim != 5:
        raise ValueError("images must have shape [N, C, D, H, W]")
    if labeled_labels.ndim != 4 or unlabeled_labels.ndim != 4:
        raise ValueError("labels must have shape [N, D, H, W]")
    if labeled_images.shape[0] == 0 or unlabeled_images.shape[0] == 0:
        raise ValueError("labeled and unlabeled batches must be nonempty")
    if labeled_labels.shape != (labeled_images.shape[0], *labeled_images.shape[-3:]):
        raise ValueError("labeled labels must match labeled images shape [N, D, H, W]")
    if unlabeled_labels.shape != (unlabeled_images.shape[0], *unlabeled_images.shape[-3:]):
        raise ValueError("unlabeled labels must match unlabeled images shape [N, D, H, W]")
    if labeled_images.shape[1:] != unlabeled_images.shape[1:]:
        raise ValueError("labeled and unlabeled image shapes must match")
    if labeled_labels.shape[1:] != unlabeled_labels.shape[1:]:
        raise ValueError("labeled and unlabeled label shapes must match")
    tensors = (labeled_labels, unlabeled_images, unlabeled_labels)
    if any(tensor.device != labeled_images.device for tensor in tensors):
        raise ValueError("all UCP tensors must be on the same device")

    pair_indices = torch.arange(
        unlabeled_images.shape[0], device=labeled_images.device
    ) % labeled_images.shape[0]
    paired_images = labeled_images[pair_indices]
    paired_labels = labeled_labels[pair_indices]
    mask = make_center_masks(unlabeled_images, scale_min, scale_max, generator)
    image_mask = mask.bool()
    label_mask = image_mask[:, 0]
    inward_images = torch.where(image_mask, paired_images, unlabeled_images)
    outward_images = torch.where(image_mask, unlabeled_images, paired_images)
    inward_labels = torch.where(label_mask, paired_labels, unlabeled_labels)
    outward_labels = torch.where(label_mask, unlabeled_labels, paired_labels)
    return inward_images, inward_labels, outward_images, outward_labels, mask


def merge_unlabeled_view(inward_probs, outward_probs, mask):
    if inward_probs.shape != outward_probs.shape:
        raise ValueError("inward and outward probabilities must have the same shape")
    if inward_probs.ndim != 5 or mask.shape != (inward_probs.shape[0], 1, *inward_probs.shape[-3:]):
        raise ValueError("mask must have shape [N, 1, D, H, W]")
    return outward_probs * mask + inward_probs * (1 - mask)


def symmetric_guidance_mask(direct_probs, merged_probs, confidence):
    if direct_probs.shape != merged_probs.shape or direct_probs.ndim != 5:
        raise ValueError("teacher probabilities must share shape [N, C, D, H, W]")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    direct_conf, direct_label = direct_probs.max(dim=1)
    merged_conf, merged_label = merged_probs.max(dim=1)
    return ((direct_label == merged_label)
            & (direct_conf >= confidence)
            & (merged_conf >= confidence))


def masked_cross_entropy(student_logits, teacher_probs, mask):
    if student_logits.shape != teacher_probs.shape or student_logits.ndim != 5:
        raise ValueError("student logits and teacher probabilities must have matching 5D shapes")
    if mask.shape != student_logits.shape[:1] + student_logits.shape[-3:]:
        raise ValueError("mask must have shape [N, D, H, W]")
    targets = teacher_probs.detach().argmax(dim=1)
    voxel_loss = F.cross_entropy(student_logits, targets, reduction="none")
    weights = mask.to(voxel_loss.dtype)
    return (voxel_loss * weights).sum() / weights.sum().clamp_min(1)


def symgd_ramp_weight(iter_num, max_iterations, max_weight):
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    progress = min(max(iter_num / max_iterations, 0.0), 1.0)
    return max_weight * (0.1 + 0.9 * progress)
