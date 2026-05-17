import pathlib

import pytest

from .client import BridgeClient


def pytest_addoption(parser):
	parser.addoption(
		"--godot",
		action="store",
		default=None,
		help="Path to the Godot binary used by the test bridge.",
	)


@pytest.fixture(scope="session")
def godot_binary(request) -> pathlib.Path:
	value = request.config.getoption("--godot")
	if not value:
		pytest.skip("--godot=<path> not provided")
	path = pathlib.Path(value)
	if not path.is_file():
		pytest.fail(f"--godot path does not exist: {path}")
	return path


@pytest.fixture
def bridge(godot_binary, tmp_path):
	log_path = tmp_path / "godot.log"
	with BridgeClient(godot_binary=godot_binary, log_path=log_path) as client:
		yield client
