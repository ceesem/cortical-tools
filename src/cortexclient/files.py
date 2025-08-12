import contextlib
import os
import re
import sys
from dataclasses import dataclass
from io import BytesIO

import cloudfiles
import pandas as pd


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@dataclass
class CloudFileViewExport:
    name: str
    size: int
    version: list[int]
    filepath_base: str
    _cf: cloudfiles.CloudFiles

    @property
    def datapath(self) -> str:
        """Get the full file path."""
        return f"{self.filepath_base}.csv.gz"

    @property
    def headerpath(self) -> str:
        """Get the header file path."""
        return f"{self.filepath_base}_header.csv"

    def _header(self) -> list:
        b = self._cf.get(self.headerpath)
        with BytesIO(b) as f:
            header = pd.read_csv(f, header=None)[0].values.tolist()
        return header

    def get_dataframe(self, version, verbose=True) -> pd.DataFrame:
        """Get the DataFrame for a specific version."""
        header = self._header()
        if verbose:
            self._cf.progress = True
            print(f"\tDownloading {self.datapath}...")
        b = self._cf.get(self.datapath)
        self._cf.progress = False
        if verbose:
            print(f"\tImporting {self.datapath}...")
        with BytesIO(b) as f:
            df = pd.read_csv(f, compression="gzip")
        df.columns = header
        return df

    def __repr__(self):
        return f"{self.name}[v{self.version}]"


class TableExportClient:
    """Client for accessing CAVE table exports at a specific cloud path.
    Talk to your dataset admin for the cloud path and to discuss what tables are made available with this route.

    Parameters
    ----------
    cloudpath : str
        The cloud path to the table exports.
    """

    def __init__(
        self,
        cloudpath: str,
    ):
        self._table_export_cloudpath = cloudpath
        self._cf = cloudfiles.CloudFiles(cloudpath, progress=False)
        self._available_files = None

    def reset_available_files(self):
        """Reset the available files cache."""
        self._available_files = None
        return self.available_files

    @property
    def available_files(self):
        """List available files at the cloudpath."""
        if self._available_files is None:
            self._available_files = []
            view_files = list(self._cf)
            view_file_sizes = self._cf.size(view_files)
            for f in view_files:
                if f.endswith(".csv.gz"):
                    name = f.split("/")[-1].replace(".csv.gz", "")
                    version_match = re.match(r"^v([0-9]*)\/.*", f)
                    if not version_match:
                        print(f"Skipping {f} as it does not match version pattern")
                        continue
                    version = int(version_match.group(1))
                    filepath_base = f.replace(".csv.gz", "")
                    self._available_files.append(
                        CloudFileViewExport(
                            name=name,
                            size=view_file_sizes[f],
                            version=version,
                            filepath_base=filepath_base,
                            _cf=self._cf,
                        )
                    )
        return self._available_files

    @property
    def available_tables(self):
        """Get the available tables."""
        return sorted(list(set([f.name for f in self.available_files])))

    def available_versions(self, table_name: str) -> list[int]:
        """Get the available versions for a specific table."""
        return sorted(
            list(set([f.version for f in self.available_files if f.name == table_name]))
        )

    def get_table(self, table_name: str, version: int) -> pd.DataFrame:
        """Download a specific table as a DataFrame."""
        if table_name not in self.available_tables:
            raise ValueError(f"Table {table_name} is not available.")
        if version not in self.available_versions(table_name):
            raise ValueError(
                f"Version {version} for table {table_name} is not available."
            )

        file = next(
            (
                f
                for f in self.available_files
                if f.name == table_name and f.version == version
            ),
            None,
        )
        if file is None:
            raise ValueError(f"No file found for {table_name} v{version}.")

        return file.get_dataframe(version)

    def available_data_df(self) -> pd.DataFrame:
        """Get a DataFrame of all available data."""
        data = []
        for file in self.available_files:
            data.append(
                {
                    "name": file.name,
                    "version": file.version,
                    "size": f"{file.size / (1024**3):.2f} Gb",
                }
            )
        return (
            pd.DataFrame(data)
            .sort_values(by=["name", "version"])
            .reset_index(drop=True)
        )
