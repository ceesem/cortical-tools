import pytest

from cortical_tools.datasets import microns_prod, microns_public, v1dd, v1dd_public


@pytest.fixture(scope="session")
def v1dd_client():
    return v1dd.client


@pytest.fixture(scope="session")
def v1dd_public_client():
    return v1dd_public.client


@pytest.fixture(scope="session")
def microns_prod_client():
    return microns_prod.client


@pytest.fixture(scope="session")
def microns_public_client():
    return microns_public.client
