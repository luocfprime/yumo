import os

import pytest


@pytest.fixture(scope="session")
def project_root(pytestconfig) -> str:
    return str(pytestconfig.rootdir)


@pytest.fixture(scope="session")
def test_data(project_root) -> str:
    return os.path.join(project_root, "tests", "data")
