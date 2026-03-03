import logging
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional C-extensions that are not available in the test
# environment (open3d, tqdm, etc.) before importing any perceptionmetrics module.
# ---------------------------------------------------------------------------
for _stub in ("open3d",):
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

# Ensure tqdm.tqdm is a callable no-op (used in detection.py)
if "tqdm" not in sys.modules:
    import tqdm  # noqa: F401 – available in the environment
_tqdm_mod = sys.modules["tqdm"]
if not callable(getattr(_tqdm_mod, "tqdm", None)):
    _tqdm_mod.tqdm = lambda iterable, **kw: iterable  # type: ignore[attr-defined]

from perceptionmetrics.datasets.yolo import build_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_YAML_TRAIN_VAL_ONLY = {
    "path": "/fake/dataset",
    "train": "images/train",
    "val": "images/val",
    # 'test' key is absent — common for many YOLO datasets
    "names": {0: "cat", 1: "dog"},
}

_FAKE_YAML_TEST_NULL = {
    "path": "/fake/dataset",
    "train": "images/train",
    "val": "images/val",
    "test": None,  # Key present but value is null (as parsed from YAML)
    "names": {0: "cat", 1: "dog"},
}

_FAKE_YAML_ALL_SPLITS = {
    "path": "/fake/dataset",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "cat", 1: "dog"},
}


def _make_patched_build_dataset(yaml_content, label_files_by_split):
    """Return a call to build_dataset with filesystem calls mocked.

    :param yaml_content: Dictionary simulating parsed YAML content
    :type yaml_content: dict
    :param label_files_by_split: Mapping of split name to list of label file paths
    :type label_files_by_split: dict
    :return: Result of build_dataset
    :rtype: tuple
    """
    fake_dataset_fname = "/fake/dataset/data.yaml"
    fake_dataset_dir = "/fake/dataset"

    def _fake_glob(pattern):
        for split, files in label_files_by_split.items():
            if split in pattern:
                return files
        return []

    def _fake_isfile(path):
        # Treat any .txt file as label, any other as image
        return True

    with patch("os.path.isfile", return_value=True), patch(
        "os.path.isdir", return_value=True
    ), patch(
        "perceptionmetrics.datasets.yolo.uio.read_yaml", return_value=yaml_content
    ), patch(
        "perceptionmetrics.datasets.yolo.glob", side_effect=_fake_glob
    ):
        return build_dataset(fake_dataset_fname, fake_dataset_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_dataset_no_test_split_no_error():
    """build_dataset must not raise TypeError when 'test' key is absent.

    Regression test for the bug where os.path.join was called with None
    when the YAML had no test split defined.

    :raises AssertionError: If a TypeError is raised or unexpected splits appear
    """
    dataset, ontology, dataset_dir = _make_patched_build_dataset(
        _FAKE_YAML_TRAIN_VAL_ONLY,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
            "val": ["/fake/dataset/labels/val/img2.txt"],
        },
    )

    assert isinstance(dataset, pd.DataFrame)
    assert "test" not in dataset["split"].values
    assert set(dataset["split"].unique()) <= {"train", "val"}


def test_build_dataset_null_test_split_no_error():
    """build_dataset must not raise TypeError when 'test' key is present but null.

    This mirrors the exact COCO8 YAML structure that triggered the original bug.

    :raises AssertionError: If a TypeError is raised or unexpected splits appear
    """
    dataset, ontology, dataset_dir = _make_patched_build_dataset(
        _FAKE_YAML_TEST_NULL,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
            "val": ["/fake/dataset/labels/val/img2.txt"],
        },
    )

    assert isinstance(dataset, pd.DataFrame)
    assert "test" not in dataset["split"].values


def test_build_dataset_missing_split_emits_warning(caplog):
    """build_dataset must log a warning for each split that is missing or null.

    :param caplog: pytest log capture fixture
    :type caplog: pytest.LogCaptureFixture
    :raises AssertionError: If no warning is logged for the missing split
    """
    with caplog.at_level(logging.WARNING, logger="root"):
        _make_patched_build_dataset(
            _FAKE_YAML_TEST_NULL,
            {
                "train": ["/fake/dataset/labels/train/img1.txt"],
                "val": ["/fake/dataset/labels/val/img2.txt"],
            },
        )

    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("test" in msg for msg in warning_messages), (
        "Expected a warning about the missing 'test' split, got: %s" % warning_messages
    )


def test_build_dataset_all_splits_present():
    """build_dataset processes all three splits when they are all defined.

    :raises AssertionError: If rows for any split are absent
    """
    dataset, ontology, dataset_dir = _make_patched_build_dataset(
        _FAKE_YAML_ALL_SPLITS,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
            "val": ["/fake/dataset/labels/val/img2.txt"],
            "test": ["/fake/dataset/labels/test/img3.txt"],
        },
    )

    assert isinstance(dataset, pd.DataFrame)
    assert set(dataset["split"].unique()) == {"train", "val", "test"}


def test_build_dataset_ontology_built_correctly():
    """build_dataset must build ontology from YAML 'names' dict.

    :raises AssertionError: If ontology keys or indices are incorrect
    """
    dataset, ontology, dataset_dir = _make_patched_build_dataset(
        _FAKE_YAML_TRAIN_VAL_ONLY,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
        },
    )

    assert "cat" in ontology
    assert "dog" in ontology
    assert ontology["cat"]["idx"] == 0
    assert ontology["dog"]["idx"] == 1
