"""Checks for experiment validity, not model-score expectations."""

import ast
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import hierarchy_experiment as h


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        return (torch.rand(3, 3, 32, 32), torch.ones(3, 1, 32, 32),
                torch.rand(3, 19), torch.stack([torch.rand(3, 11), torch.zeros(3, 11)], -1))

    def test_reference_models_match_original_h3(self):
        tree = ast.parse((h.BASE / "h3.py").read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name in {"masked_mean", "Readout", "BatteryModel"}]
        scope = {"torch": torch, "nn": nn, "F": F, "math": math, "Q_DIM": 19, "D": 64, "GAS": h.GAS}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "h3 definitions", "exec"), scope)
        inputs = self.inputs()
        for mode in ("flat", "conditional"):
            h.seed_everything(42)
            original = scope["BatteryModel"](mode).eval()
            h.seed_everything(42)
            replacement = h.BatteryHierarchyModel(mode).eval()
            for name, weight in original.state_dict().items():
                self.assertTrue(torch.equal(weight, replacement.state_dict()[name]), name)
            with torch.no_grad():
                torch.testing.assert_close(original(*inputs), replacement(*inputs)[0], rtol=0, atol=0)

    def test_hierarchy_has_no_direct_sensor_bypass(self):
        model = h.BatteryHierarchyModel("hierarchy_supervised", d=16).eval()
        inputs = self.inputs()
        hooks = [module.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
                 for module in (model.gas_predictor, model.image_predictor)]
        with torch.no_grad():
            first = model(*inputs)[0]
            second = model(inputs[0], inputs[1], inputs[2] + 100, inputs[3])[0]
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_auxiliary_loss_ignores_missing_values(self):
        prediction = torch.tensor([[2.0, 3.0]], requires_grad=True)
        target = torch.tensor([[1.0, float("nan")]])
        sse, count = h.masked_squared_error(prediction, target, torch.tensor([[True, False]]))
        self.assertEqual(float(sse.detach()), 1)
        self.assertEqual(int(count), 1)
        sse.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[2.0, 0.0]]))

    def test_all_missing_observations_are_finite(self):
        inputs = list(self.inputs())
        inputs[1].zero_()
        inputs[3][:, :, 1] = 1
        for mode in h.MODES:
            model = h.BatteryHierarchyModel(mode, d=16)
            logits, gas_prediction, image_prediction = model(*inputs)
            self.assertTrue(torch.isfinite(logits).all())
            gl, il = h.auxiliary_loss(gas_prediction, image_prediction, inputs[3],
                                    torch.zeros_like(image_prediction), torch.zeros_like(image_prediction, dtype=torch.bool))
            self.assertEqual(float((gl + il).detach()), 0)

    def test_fixed_image_targets_exclude_padding(self):
        rgb = np.full((3, 8, 8), 255, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        rgb[:, :4, :4] = 128
        mask[:4, :4] = 1
        target, valid = h.image_descriptor(rgb, mask, grid=2)
        self.assertEqual(int(valid.sum()), 3)
        np.testing.assert_allclose(target[valid], 128 / 255, rtol=1e-6)
        np.testing.assert_array_equal(target[~valid], 0)

    def test_group_holdout_and_fit_only_normalization(self):
        def make_frame(groups, prefix):
            n = groups * 6
            frame = pd.DataFrame({name: np.arange(n, dtype=float) for name in h.NUMERIC})
            frame["sample_id"] = [f"{prefix}_sample_{i}" for i in range(n)]
            frame["experiment_id"] = [f"{prefix}_experiment_{i // 6}" for i in range(n)]
            frame["label"] = np.tile(np.arange(1, 7), groups)
            frame["bat_class"], frame["tr_method"], frame["soc_status"], frame["bat_cc"] = 1, 1, 1, 10.0
            return frame
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in (("train", make_frame(10, "train")), ("validation", make_frame(2, "val"))):
                (root / name).mkdir()
                frame.to_csv(root / name / f"{name}.csv", index=False)
            frames = h.split_frames(root, 7, 0.2)
        groups = [set(frame.experiment_id) for frame in frames.values()]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])
        prep = h.Preprocessor(frames["fit"])
        before = prep.metadata()
        shifted = frames["validation"].copy()
        shifted[h.NUMERIC] += 1e6
        prep.transform(shifted)
        self.assertEqual(before, prep.metadata())
        self.assertAlmostEqual(prep.mean["meas_v"], frames["fit"].meas_v.mean())


if __name__ == "__main__":
    unittest.main(verbosity=2)
