import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "point_global_gradient_scale_control_test",
    ROOT / "scripts/run_point_global_gradient_scale_control.py")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_scale_multiplies_only_learned_gradient_and_keeps_zero_zero():
    learned = torch.tensor([[3.0, 4.0], [0.0, 0.0], [-2.0, 1.0]])
    scaled = experiment.scale_gradient(learned, 2.5)
    assert torch.equal(scaled, learned * 2.5)
    assert torch.equal(experiment.scale_gradient(learned, 1.0), learned)
    assert torch.equal(scaled[1], torch.zeros(2))
    assert torch.allclose(experiment.scale_gradient(learned, 2.5).norm(dim=-1),
                          learned.norm(dim=-1) * 2.5)


@pytest.mark.parametrize("c", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_global_scales_fail(c):
    with pytest.raises(ValueError):
        experiment.scale_gradient(torch.ones((1, 2)), c)


def test_projection_preserves_signed_directional_derivative():
    learned = torch.tensor([[3.0, 4.0], [3.0, 4.0], [0.0, 0.0]])
    reference = torch.tensor([[0.0, 10.0], [-3.0, -4.0], [5.0, 0.0]])
    projected = experiment.projection_gradient(learned, reference)
    # d_L=(.6,.8); first dot is 8, second is -5; sign is not clamped.
    assert torch.allclose(projected[0], torch.tensor([4.8, 6.4]))
    assert torch.allclose(projected[1], torch.tensor([-3.0, -4.0]))
    assert torch.equal(projected[2], torch.zeros(2))


def test_case_boundaries_are_predeclared():
    base_rows = [
        {"c": 1.0, "success_rate": .50, "collision_rate": .50,
         "discounted_return": 0., "episodes": 4096},
        {"c": 1.5, "success_rate": .55, "collision_rate": .45,
         "discounted_return": .1, "episodes": 4096},
        {"c": 2.0, "success_rate": .60, "collision_rate": .40,
         "discounted_return": .2, "episodes": 4096},
        {"c": 2.5, "success_rate": .65, "collision_rate": .35,
         "discounted_return": .3, "episodes": 4096},
        {"c": 3.0, "success_rate": .70, "collision_rate": .30,
         "discounted_return": .4, "episodes": 4096},
        {"c": 4.0, "success_rate": .75, "collision_rate": .25,
         "discounted_return": .5, "episodes": 4096},
    ]
    # Create a tiny temporary baseline-like result bundle without modifying repo artifacts.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "final-independent.csv"
        import pandas as pd
        pd.DataFrame([
            {"method": "learned", "success_rate": .50},
            {"method": "magnitude_only", "success_rate": .70},
            {"method": "full_reference", "success_rate": .75},
        ]).to_csv(path, index=False)
        old_dir = experiment.BASELINE_DIR
        experiment.BASELINE_DIR = Path(directory)
        try:
            _, result = experiment.compute_global_results(base_rows, {"beta": 4.0})
        finally:
            experiment.BASELINE_DIR = old_dir
    assert result["best_global_scale"] == 4.0
    assert result["recovery_of_magnitude_only"] == pytest.approx(1.25)
    assert result["recovery_of_full_reference"] == pytest.approx(1.0)
    assert result["case"] == "A"


@pytest.mark.parametrize(
    ("best_success", "expected_case"),
    [(.60, "B"), (.54, "C")],
)
def test_partial_and_small_recovery_follow_frozen_projection_rule(best_success, expected_case):
    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    rows = [{"c": c, "success_rate": .50 if c == 1.0 else best_success,
             "collision_rate": 1 - (.50 if c == 1.0 else best_success),
             "discounted_return": 0.0, "episodes": 4096} for c in scales]
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        import pandas as pd
        pd.DataFrame([
            {"method": "learned", "success_rate": .50},
            {"method": "magnitude_only", "success_rate": .70},
            {"method": "full_reference", "success_rate": .75},
        ]).to_csv(Path(directory) / "final-independent.csv", index=False)
        old_dir = experiment.BASELINE_DIR
        experiment.BASELINE_DIR = Path(directory)
        try:
            _, result = experiment.compute_global_results(rows, {"beta": 4.0})
        finally:
            experiment.BASELINE_DIR = old_dir
    assert result["case"] == expected_case
    assert result["projection_required"] is True


def test_reference_estimator_guard_fails_closed_and_restores_functions():
    original_batch = experiment.experiment.mc_reference_batch
    original_independent = experiment.experiment.longrun.independent_mc
    calls_before = experiment.REFERENCE_ACCESS_CALLS
    with pytest.raises(RuntimeError, match="forbidden"):
        with experiment.block_reference_estimators():
            experiment.experiment.mc_reference_batch(None)
    assert experiment.experiment.mc_reference_batch is original_batch
    assert experiment.experiment.longrun.independent_mc is original_independent
    assert experiment.REFERENCE_ACCESS_CALLS == calls_before + 1
    experiment.REFERENCE_ACCESS_CALLS = calls_before
