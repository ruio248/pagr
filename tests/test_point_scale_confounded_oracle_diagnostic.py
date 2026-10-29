import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "point_scale_confounded_oracle_diagnostic_test",
    ROOT / "scripts/run_point_scale_confounded_oracle_diagnostic.py")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_projection_scale_and_residual_match_closed_form():
    learned = torch.tensor([1., 2., -1.])
    reference = torch.tensor([2., 2., 0.])
    scale = experiment.projection_scale(learned, reference)
    expected = float(torch.dot(learned, reference) / torch.dot(learned, learned))
    assert scale == pytest.approx(expected)
    metrics = experiment.projection_metrics(learned, reference, "q")
    residual = reference-scale*learned
    assert metrics["q_relative_residual"] == pytest.approx(
        float(residual.norm()/reference.norm()))
    assert abs(float(torch.dot(learned, residual))) < 1e-6


def test_projection_preserves_negative_scale_when_directions_are_opposed():
    source = torch.tensor([1., -2.])
    target = -3*source
    assert experiment.projection_scale(source, target) == pytest.approx(-3.)
    assert experiment.projection_metrics(source, target, "x")["x_relative_residual"] < 1e-7


def test_projection_rejects_shape_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError):
        experiment.projection_scale(torch.ones(2), torch.ones(3))
    with pytest.raises(FloatingPointError):
        experiment.projection_scale(torch.tensor([float("nan")]), torch.ones(1))
