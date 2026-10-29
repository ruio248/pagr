import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "point_gradient_magnitude_direction_test",
    ROOT / "scripts/run_point_gradient_magnitude_direction.py")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_gradient_decomposition_preserves_the_prescribed_component_and_zero_rule():
    learned = torch.tensor([[3., 4.], [0., 0.], [2., 0.], [1., 1.]])
    reference = torch.tensor([[0., 10.], [5., 0.], [0., 0.], [-1., 1.]])
    labels = experiment.decompose_labels(learned, reference, 1e-8)

    assert torch.equal(labels["learned"], learned)
    assert torch.equal(labels["full_reference"], reference)
    # Nonzero learned direction, reference magnitude.
    assert torch.allclose(labels["magnitude_only"][0], torch.tensor([6., 8.]))
    assert torch.allclose(labels["magnitude_only"].norm(dim=-1)[[0, 2, 3]],
                          reference.norm(dim=-1)[[0, 2, 3]])
    assert torch.allclose(experiment.cosine_rows(labels["magnitude_only"], learned)[[0, 3]],
                          torch.ones(2), atol=1e-6)
    # Learned magnitude, reference direction.
    assert torch.allclose(labels["direction_only"][0], torch.tensor([0., 5.]))
    assert torch.allclose(labels["direction_only"].norm(dim=-1)[[0, 1, 3]],
                          learned.norm(dim=-1)[[0, 1, 3]])
    assert torch.allclose(experiment.cosine_rows(labels["direction_only"], reference)[[0, 3]],
                          torch.ones(2), atol=1e-6)
    # Zero learned/reference gradients are not assigned an arbitrary direction.
    assert torch.equal(labels["magnitude_only"][1], torch.zeros(2))
    assert torch.equal(labels["direction_only"][1], torch.zeros(2))
    assert torch.equal(labels["magnitude_only"][2], torch.zeros(2))
    assert torch.equal(labels["direction_only"][2], torch.zeros(2))


def test_gradient_metrics_keep_zero_vectors_and_relative_errors_explicit():
    learned = torch.tensor([[1., 0.], [0., 0.]])
    reference = torch.tensor([[0., 2.], [0., 0.]])
    labels = experiment.decompose_labels(learned, reference)
    rows, summary = experiment.vector_metrics(learned, reference, labels, 1e-8)

    assert len(rows) == 2
    assert summary["learned_zero_fraction"] == .5
    assert summary["reference_zero_fraction"] == .5
    assert rows[1]["learned_zero"] and rows[1]["reference_zero"]
    assert summary["relative_error_full_reference"] == 0.
    assert abs(summary["relative_error_direction_only"] - .25) < 1e-6
    assert abs(summary["learned_to_reference_norm_ratio"] - .5) < 1e-6
    assert abs(rows[0]["sample_relative_error_direction_only"] - .5) < 1e-6
    assert rows[1]["sample_relative_error_direction_only"] == 0.


def test_restore_rowwise_errors_from_saved_norms_and_cosine():
    learned = torch.tensor([[3., 4.], [0., 0.]])
    reference = torch.tensor([[0., 10.], [0., 0.]])
    labels = experiment.decompose_labels(learned, reference)
    original, _ = experiment.vector_metrics(learned, reference, labels, 1e-8)
    legacy = [{key: row[key] for key in (
        "learned_gradient_norm", "reference_gradient_norm", "learned_reference_cosine")}
        for row in original]

    restored = experiment.restore_sample_error_metrics(legacy, 1e-8, 1e-8)
    for row, expected in zip(restored, original):
        assert abs(row["learned_to_reference_norm_ratio"] - .5) < 1e-6
        for name in ("learned", "magnitude_only", "direction_only", "full_reference"):
            assert abs(row[f"sample_relative_error_{name}"] -
                       expected[f"sample_relative_error_{name}"]) < 1e-6
