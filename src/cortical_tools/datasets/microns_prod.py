from standard_transform.datasets import minnie_ds

from ..common import DatasetClient


class MicronsProdClient(DatasetClient):
    datastack_name = "minnie65_phase3_v1"
    server_address = "https://global.daf-apis.com"
    _cell_id_lookup_view = "nucleus_detection_lookup_v1"
    _root_id_lookup_main_table = "nucleus_detection_v0"
    _root_id_lookup_alt_tables = ["nucleus_alternative_points"]

    def __init__(self):
        super().__init__(
            datastack_name=self.datastack_name,
            server_address=self.server_address,
            cell_id_lookup_view=self._cell_id_lookup_view,
            root_id_lookup_main_table=self._root_id_lookup_main_table,
            root_id_lookup_alt_tables=self._root_id_lookup_alt_tables,
            dataset_transform=minnie_ds,
        )

    def __repr__(self):
        return f"MicronsPublicClient(datastack_name={self.datastack_name}, version={self.cave.materialize.version})"

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Necessary for IPython to detect _repr_html_ for subclasses."""
        return {"text/html": self.__repr_html__()}, {}


client = MicronsProdClient()
