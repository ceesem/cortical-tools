import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import DatasetClient

__version__ = "0.0.5"


def load_client(dataset_name: str) -> "DatasetClient":
    """
    Dynamically load a dataset client by name.

    Args:
        dataset_name: Name of the dataset. Valid options are:
            - 'v1dd'
            - 'v1dd_public'
            - 'microns_prod'
            - 'microns_public'

    Returns:
        The pre-instantiated client for the specified dataset.

    Raises:
        ValueError: If dataset_name is not recognized.
        ImportError: If the dataset module cannot be imported.
    """
    valid_datasets = {"v1dd", "v1dd_public", "microns_prod", "microns_public"}

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid options are: {', '.join(sorted(valid_datasets))}"
        )

    try:
        module = importlib.import_module(f"cortical_tools.datasets.{dataset_name}")
        return module.client
    except ImportError as e:
        raise ImportError(f"Failed to import dataset '{dataset_name}': {e}") from e
