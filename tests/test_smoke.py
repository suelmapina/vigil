"""Smoke tests: the package and its dependencies import cleanly."""

import importlib

import pytest

import vigil
from vigil import config


def test_version():
    assert vigil.version


def test_paths_inside_repo():
    assert config.rawdir.is_relative_to(config.root)
    assert config.processeddir.is_relative_to(config.root)


def test_split_is_temporal():
    assert max(config.trainmonths) < min(config.testmonths)


@pytest.mark.parametrize("module", ["lightgbm", "mlflow", "fastapi", "streamlit", "kagglehub"])
def test_dependencies_import(module):
    importlib.import_module(module)
