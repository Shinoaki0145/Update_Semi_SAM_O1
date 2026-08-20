import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn


CODE_DIR = Path(__file__).resolve().parents[1]


def load_training_module():
    tensorboard = types.ModuleType("tensorboardX")
    tensorboard.SummaryWriter = object
    transforms = types.ModuleType("torchvision.transforms")
    transforms.Compose = lambda items: items
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = transforms
    modules = {
        "tensorboardX": tensorboard,
        "torchvision": torchvision,
        "torchvision.transforms": transforms,
    }
    argv = [
        "train_SemiSAM_O1.py",
        "--backbone", "mt",
        "--ucp_symgd",
        "--ucp_start_round", "2",
        "--ucp_scale_min", "0.5",
        "--ucp_scale_max", "0.5",
        "--symgd_confidence", "0.6",
        "--symgd_weight", "1.0",
    ]
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location(
        "train_semisam_o1_test", CODE_DIR / "train_SemiSAM_O1.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
        spec.loader.exec_module(module)
    return module


class TinySegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return torch.cat((-images, images), dim=1)


class TrainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_training_module()

    def test_cli_accepts_approved_ucp_symgd_arguments(self):
        args = self.train.args
        self.assertTrue(args.ucp_symgd)
        self.assertEqual(args.ucp_start_round, 2)
        self.assertEqual(args.ucp_scale_min, 0.5)
        self.assertEqual(args.ucp_scale_max, 0.5)
        self.assertEqual(args.symgd_confidence, 0.6)
        self.assertEqual(args.symgd_weight, 1.0)

    def test_round_before_start_adds_no_forward_pass(self):
        student = TinySegmentor()
        teacher = TinySegmentor()
        volume = torch.linspace(-1, 1, 128).reshape(2, 1, 4, 4, 4)
        labels = (volume[:, 0] > 0).long()
        direct_student = student(volume)[1:]
        direct_teacher = teacher(volume[1:])
        student.calls = teacher.calls = 0

        ucp, sym, kept, gamma = self.train._compute_ucp_symgd_losses(
            self.train.args, 1, 0, student, teacher, volume, labels,
            direct_student, direct_teacher, nn.CrossEntropyLoss(),
            self.train.losses.DiceLoss(2),
        )

        self.assertEqual(student.calls, 0)
        self.assertEqual(teacher.calls, 0)
        self.assertEqual((ucp.item(), sym.item(), kept, gamma), (0.0, 0.0, 0.0, 0.0))

    def test_active_round_computes_losses_with_one_mixed_forward_per_model(self):
        torch.manual_seed(7)
        student = TinySegmentor()
        teacher = TinySegmentor()
        volume = torch.linspace(-1, 1, 128).reshape(2, 1, 4, 4, 4).requires_grad_()
        labels = (volume.detach()[:, 0] > 0).long()
        direct_student = student(volume)[1:]
        direct_student.retain_grad()
        direct_teacher = teacher(volume[1:])
        student.calls = teacher.calls = 0
        args = types.SimpleNamespace(**vars(self.train.args))
        args.max_iterations = 10

        ucp, sym, kept, gamma = self.train._compute_ucp_symgd_losses(
            args, 2, 5, student, teacher, volume, labels,
            direct_student, direct_teacher, nn.CrossEntropyLoss(),
            self.train.losses.DiceLoss(2),
        )

        self.assertEqual(student.calls, 1)
        self.assertEqual(teacher.calls, 1)
        self.assertTrue(torch.isfinite(ucp))
        self.assertTrue(torch.isfinite(sym))
        self.assertGreaterEqual(kept, 0.0)
        self.assertLessEqual(kept, 1.0)
        self.assertAlmostEqual(gamma, 0.55)
        (ucp + sym).backward()
        self.assertIsNotNone(direct_student.grad)


if __name__ == "__main__":
    unittest.main()
