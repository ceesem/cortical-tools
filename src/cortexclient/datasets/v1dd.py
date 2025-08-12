from standard_transform.datasets import v1dd_ds

from ..common import DatasetClient


class V1ddClient(DatasetClient):
    datastack_name = "v1dd"
    server_address = "https://global.em.brain.allentech.org"
    _cell_id_lookup_view = "nucleus_alternative_lookup"
    _root_id_lookup_main_table = "nucleus_detection_v0"
    _root_id_lookup_alt_tables = ["nucleus_alternative_points"]

    def __init__(self):
        super().__init__(
            datastack_name=self.datastack_name,
            server_address=self.server_address,
            cell_id_lookup_view=self._cell_id_lookup_view,
            root_id_lookup_main_table=self._root_id_lookup_main_table,
            root_id_lookup_alt_tables=self._root_id_lookup_alt_tables,
            dataset_transform=v1dd_ds,
        )

    def __repr__(self):
        return f"V1ddClient(datastack_name={self.datastack_name}, version={self.cave.materialize.version})"

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Necessary for IPython to detect _repr_html_ for subclasses."""
        return {"text/html": self.__repr_html__()}, {}


client = V1ddClient()
