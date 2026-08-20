import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ucp_symgd import (  # noqa: E402
    make_center_masks,
    masked_cross_entropy,
    merge_unlabeled_view,
    symgd_ramp_weight,
    symmetric_guidance_mask,
    ucp_symgd_enabled,
    unified_copy_paste,
    validate_ucp_symgd_config,
)


class UcpSymgdTests(unittest.TestCase):
    def test_center_mask_has_expected_fixed_scale_geometry(self):
        images = torch.zeros(2, 1, 4, 6, 8)
        mask = make_center_masks(images, 0.5, 0.5)
        expected = torch.zeros_like(mask)
        expected[:, :, 1:3, 1:4, 2:6] = 1
        torch.testing.assert_close(mask, expected)

    def test_copy_paste_is_symmetric_and_cycles_labeled_samples(self):
        labeled_images = torch.stack((
            torch.full((1, 4, 4, 4), 10.0),
            torch.full((1, 4, 4, 4), 20.0),
        ))
        labeled_labels = torch.stack((
            torch.full((4, 4, 4), 7),
            torch.full((4, 4, 4), 8),
        ))
        unlabeled_images = torch.stack([
            torch.full((1, 4, 4, 4), float(value)) for value in (1, 2, 3)
        ])
        unlabeled_labels = torch.stack([
            torch.full((4, 4, 4), value) for value in (1, 2, 3)
        ])

        u_in, q_in, u_out, q_out, mask = unified_copy_paste(
            labeled_images,
            labeled_labels,
            unlabeled_images,
            unlabeled_labels,
            0.5,
            0.5,
        )

        center = mask.bool().expand_as(u_in)
        label_center = mask[:, 0].bool()
        paired_images = labeled_images[torch.tensor([0, 1, 0])]
        paired_labels = labeled_labels[torch.tensor([0, 1, 0])]
        torch.testing.assert_close(u_in, torch.where(center, paired_images, unlabeled_images))
        torch.testing.assert_close(u_out, torch.where(center, unlabeled_images, paired_images))
        torch.testing.assert_close(q_in, torch.where(label_center, paired_labels, unlabeled_labels))
        torch.testing.assert_close(q_out, torch.where(label_center, unlabeled_labels, paired_labels))

    def test_merged_view_reconstructs_unlabeled_regions(self):
        inward = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
        outward = inward + 100
        mask = torch.zeros(1, 1, 2, 3, 4)
        mask[:, :, :, 1:, 1:3] = 1
        merged = merge_unlabeled_view(inward, outward, mask)
        expected = torch.where(mask.bool(), outward, inward)
        torch.testing.assert_close(merged, expected)

    def test_guidance_mask_keeps_only_confident_agreement(self):
        direct = torch.tensor([[
            [[[0.90, 0.10, 0.20]]],
            [[[0.05, 0.85, 0.20]]],
            [[[0.05, 0.05, 0.60]]],
        ]])
        merged = torch.tensor([[
            [[[0.85, 0.90, 0.15]]],
            [[[0.10, 0.05, 0.15]]],
            [[[0.05, 0.05, 0.70]]],
        ]])
        mask = symmetric_guidance_mask(direct, merged, 0.8)
        expected = torch.tensor([[[[True, False, False]]]])
        torch.testing.assert_close(mask, expected)

    def test_empty_mask_returns_differentiable_zero(self):
        logits = torch.randn(1, 3, 1, 1, 2, requires_grad=True)
        teacher = torch.full((1, 3, 1, 1, 2), 1 / 3)
        mask = symmetric_guidance_mask(teacher, teacher, 0.95)
        loss = masked_cross_entropy(logits, teacher, mask)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))

    def test_masked_cross_entropy_is_multiclass_and_backpropagates(self):
        logits = torch.tensor([[
            [[[3.0, 0.0]]],
            [[[0.0, 2.0]]],
            [[[0.0, 0.0]]],
        ]], requires_grad=True)
        teacher = torch.tensor([[
            [[[0.9, 0.1]]],
            [[[0.05, 0.8]]],
            [[[0.05, 0.1]]],
        ]])
        mask = torch.tensor([[[[True, False]]]])
        loss = masked_cross_entropy(logits, teacher, mask)
        expected = F.cross_entropy(logits[:, :, :, :, :1], torch.zeros(1, 1, 1, 1, dtype=torch.long))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_activation_and_weight_schedule(self):
        self.assertFalse(ucp_symgd_enabled(False, 3, 2))
        self.assertFalse(ucp_symgd_enabled(True, 1, 2))
        self.assertTrue(ucp_symgd_enabled(True, 2, 2))
        self.assertAlmostEqual(symgd_ramp_weight(0, 100, 1.0), 0.1)
        self.assertAlmostEqual(symgd_ramp_weight(50, 100, 1.0), 0.55)
        self.assertAlmostEqual(symgd_ramp_weight(100, 100, 1.0), 1.0)

    def test_invalid_configuration_is_rejected(self):
        invalid = (
            ("mt", 0, 0.3, 0.6, 0.95, 1.0),
            ("mt", 2, 0.0, 0.6, 0.95, 1.0),
            ("mt", 2, 0.7, 0.6, 0.95, 1.0),
            ("mt", 2, 0.3, 1.1, 0.95, 1.0),
            ("mt", 2, 0.3, 0.6, 1.1, 1.0),
            ("mt", 2, 0.3, 0.6, 0.95, -1.0),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_ucp_symgd_config(*values)
        with self.assertRaises(ValueError):
            validate_ucp_symgd_config("dan", 2, 0.3, 0.6, 0.95, 1.0, enabled=True)


if __name__ == "__main__":
    unittest.main()
