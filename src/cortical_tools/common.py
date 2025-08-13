import datetime
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from meshparty.meshwork import Meshwork

import numpy as np
import numpy.typing as npt
import pandas as pd
import pcg_skel
import standard_transform
from caveclient import CAVEclient
from caveclient.frameworkclient import CAVEclientFull
from nglui import statebuilder as sb

from .files import TableExportClient
from .mesh import MeshClient


def null_function_factory(arguments_to_set=[]):
    def null_function(*args, **kwargs):
        """
        A placeholder function that does nothing.
        """
        raise NotImplementedError(
            "This function is a placeholder. Arguments must be set in the main class: {}".format(
                arguments_to_set
            )
        )


def cell_id_to_root_id_factory(
    default_datastack_name,
    default_server_address,
    lookup_view_name,
):
    def cell_id_to_root_id(
        cell_ids: list[int],
        client: Optional[CAVEclient] = None,
        timestamp: Optional[datetime.datetime] = None,
        materialization_version: Optional[int] = None,
        filter_empty: bool = True,
    ) -> npt.NDArray:
        """
        Convert cell IDs to root IDs using the CAVEclient.

        Parameters
        ----------
        cell_ids : list[int]
            List of cell IDs to convert.
        client : CAVEclient, optional
            CAVEclient instance, by default None.
        timestamp : datetime.datetime, optional
            Timestamp for the query, by default current time.
        materialization_version : int, optional
            Materialization version, by default None.

        Returns
        -------
        pd.Series
            Series containing the root IDs with cell IDs as index.
            Cell ids that do not map to a root id will have NaN values.
        """
        if client is None:
            client = CAVEclient(
                datastack_name=default_datastack_name,
                server_address=default_server_address,
            )
        view_name = lookup_view_name
        nuc_df = (
            client.materialize.views[view_name](
                id=cell_ids,
            )
            .query(
                split_positions=True, materialization_version=materialization_version
            )
            .set_index("id")
        )

        if timestamp is not None:
            nuc_df["pt_root_id"] = client.chunkedgraph.get_roots(
                nuc_df["pt_supervoxel_id"], timestamp=timestamp
            )

        cell_id_df = pd.DataFrame(
            index=cell_ids,
        )

        cell_id_df = cell_id_df.merge(
            nuc_df[["pt_root_id"]],
            left_index=True,
            right_index=True,
            how="inner",
        ).sort_index()
        add_back_index = np.setdiff1d(cell_ids, cell_id_df.index.values)
        if len(add_back_index) > 0 and not filter_empty:
            cell_id_df = pd.concat(
                [
                    cell_id_df,
                    pd.DataFrame(index=add_back_index, data={"pt_root_id": -1}),
                ]
            )
            cell_id_df = cell_id_df.loc[cell_ids]
        return cell_id_df["pt_root_id"].rename("root_id")

    return cell_id_to_root_id


def _selective_lookup(
    query_idx: pd.Index,
    client: CAVEclient,
    timestamp: datetime.datetime,
    main_table: str,
    alt_tables: list,
):
    lookup_df_main = client.materialize.tables[main_table](
        pt_root_id=query_idx.values
    ).live_query(timestamp=timestamp, split_positions=True)
    lookup_df_main = lookup_df_main[["id", "pt_root_id"]].set_index("pt_root_id")

    if len(lookup_df_main) < len(query_idx):
        lookup_df_alts = []
        for alt_table in alt_tables:
            lookup_df_alt = client.materialize.tables[alt_table](
                pt_ref_root_id=query_idx.values,
            ).live_query(timestamp=timestamp, split_positions=True)
            if len(lookup_df_alt) > 0:
                lookup_df_alts.append(lookup_df_alt[["id", "pt_root_id"]])
        if len(lookup_df_alts) > 0:
            lookup_df_alt_concat = pd.concat(lookup_df_alts)[
                ["id", "pt_root_id"]
            ].set_index("pt_root_id")
        else:
            lookup_df_alt_concat = pd.DataFrame()

        lookup_df = pd.concat([lookup_df_main, lookup_df_alt_concat])
    else:
        lookup_df = lookup_df_main
    return lookup_df


def root_id_to_cell_id_factory(
    default_datastack_name: str,
    default_server_address: str,
    main_table: str,
    alt_tables: list[str],
):
    def root_id_to_cell_id(
        root_ids: list[int],
        client: Optional[CAVEclient] = None,
        filter_empty: bool = False,
    ):
        """
        Lookup the cell id for a list of root ids in the microns dataset
        """
        if client is None:
            client = CAVEclient(
                datastack_name=default_datastack_name,
                server_address=default_server_address,
            )

        root_ids = np.unique(root_ids)
        all_cell_df = pd.DataFrame(
            index=root_ids,
            data={"cell_id": -1, "done": False},
        )
        all_cell_df["cell_id"] = all_cell_df["cell_id"].astype(int)
        earliest_timestamp = client.chunkedgraph.get_root_timestamps(
            root_ids, latest=False
        )
        latest_timestamp = client.chunkedgraph.get_root_timestamps(
            root_ids, latest=True
        )
        all_cell_df["ts0"] = earliest_timestamp
        all_cell_df["ts1"] = latest_timestamp

        while not np.all(all_cell_df["done"].values):
            ts = all_cell_df.query("done == False").ts1.iloc[0]
            qry_idx = all_cell_df[
                (all_cell_df.ts0 < ts) & (all_cell_df.ts1 >= ts)
            ].index

            lookup_df = _selective_lookup(
                qry_idx,
                client,
                ts,
                main_table=main_table,
                alt_tables=alt_tables,
            )

            # Update the pt root ids of found cells, but the done status of all queried cells
            all_cell_df.loc[lookup_df.index, "cell_id"] = lookup_df["id"].astype(int)
            all_cell_df.loc[qry_idx, "done"] = True

        if filter_empty:
            all_cell_df = all_cell_df.query("cell_id != -1")
        return all_cell_df["cell_id"].astype(int).loc[root_ids]

    return root_id_to_cell_id


class DatasetClient:
    def __init__(
        self,
        datastack_name: Optional[str] = None,
        server_address: Optional[str] = None,
        caveclient: Optional[CAVEclient] = None,
        *,
        materialization_version: Optional[int] = None,
        cell_id_lookup_view: Optional[str] = None,
        root_id_lookup_main_table: Optional[str] = None,
        root_id_lookup_alt_tables: Optional[list[str]] = None,
        dataset_transform: Optional[standard_transform.datasets.Dataset] = None,
        static_table_cloudpath: Optional[str] = None,
    ):
        if caveclient is None:
            caveclient = CAVEclient(
                datastack_name=datastack_name,
                server_address=server_address,
                version=materialization_version,
            )
        else:
            datastack_name = caveclient.datastack_name
            server_address = caveclient.server_address

        self._client = caveclient
        self._client.version = (
            int(caveclient.materialize.version)
            if materialization_version is None
            else int(materialization_version)
        )

        self._datastack_name = datastack_name
        self._server_address = server_address

        if cell_id_lookup_view is None:
            self.cell_id_to_root_id = null_function_factory(
                arguments_to_set=["cell_id_lookup_view"]
            )
        else:
            self.cell_id_to_root_id = cell_id_to_root_id_factory(
                default_datastack_name=datastack_name,
                default_server_address=server_address,
                lookup_view_name=cell_id_lookup_view,
            )

        if root_id_lookup_main_table is None:
            self.root_id_to_cell_id = null_function_factory(
                arguments_to_set=["root_id_lookup_main_table"]
            )
        else:
            self.root_id_to_cell_id = root_id_to_cell_id_factory(
                default_datastack_name=datastack_name,
                default_server_address=server_address,
                main_table=root_id_lookup_main_table,
                alt_tables=root_id_lookup_alt_tables,
            )

        if dataset_transform is None:
            self._dataset_transform = null_function_factory(
                arguments_to_set=["dataset_transform"]
            )
        else:
            self._dataset_transform = dataset_transform

        self._mesh_client = MeshClient(caveclient=caveclient)

        self.tables = self.cave.materialize.tables
        self.views = self.cave.materialize.views

        if static_table_cloudpath is None:
            self.exports = null_function_factory(
                arguments_to_set=["static_table_cloudpath"]
            )
        else:
            self.exports = TableExportClient(static_table_cloudpath)

    def set_export_cloudpath(self, cloudpath: str):
        """
        Set the cloud path for static table exports.
        """
        self.exports = TableExportClient(cloudpath)

    @property
    def cave(self) -> CAVEclientFull:
        """
        Get the CAVEclient instance for this CortexClient.
        """
        return self._client

    @property
    def datastack_name(self) -> str:
        """
        Get the name of the datastack associated with this CortexClient.
        """
        return self._datastack_name

    @property
    def server_address(self) -> str:
        """
        Get the server address associated with this CortexClient.
        """
        return self._server_address

    @property
    def dataset_transform(self) -> standard_transform.datasets.Dataset:
        """
        Get the dataset transform associated with this CortexClient.
        """
        return self._dataset_transform

    @property
    def mesh(self) -> MeshClient:
        """
        Get the MeshClient instance for this CortexClient.
        """
        return self._mesh_client

    @property
    def space(self) -> standard_transform.datasets.Dataset:
        """
        Get the dataset transform for this CortexClient.
        """
        return self._dataset_transform

    @property
    def version(self) -> int:
        """
        Get the materialization version of the CAVEclient.
        """
        return self.cave.materialize.version

    @version.setter
    def version(self, value: int):
        """
        Set the materialization version of the CAVEclient.
        """
        self.cave.materialize.version = value

    def get_l2_ids(self, root_id: int) -> np.ndarray:
        """Get level 2 ids for a root id.

        Parameters
        ----------
        root_id : int
            Root ID for a neuron

        Returns
        -------
        list[int]
            List of level 2 IDs for the specified root ID.
        """
        return self.cave.chunkedgraph.get_roots(root_id, stop_layer=2)

    def get_skeleton(
        self,
        root_id: int,
        synapses: bool = True,
        restore_graph: bool = False,
        restore_properties: bool = True,
        synapse_reference_tables: Optional[dict] = None,
        skeleton_version: Optional[int] = None,
        transform: Optional[Literal["rigid", "streamline"]] = None,
    ) -> "Meshwork":
        """
        Get the meshwork for a specific root ID.

        Parameters
        ----------
        root_id : int
            Root ID for a neuron
        synapses : bool, optional
            If True, include synapses in the meshwork, by default True
        restore_graph : bool, optional
            If True, restore the graph structure, by default False
        restore_properties : bool, optional
            If True, restore the properties of the meshwork, by default True
        synapse_reference_tables : dict, optional
            Additional synapse reference tables to use, by default None
        skeleton_version : int, optional
            Version of the skeleton to use, by default None
        transform : Literal["rigid", "streamline"], optional
            Type of transformation to apply, by default None

        Returns
        -------
        Meshwork
            The meshwork for the specified root ID.
        """
        if skeleton_version is None:
            skeleton_version = 4
        nrn = pcg_skel.get_meshwork_from_client(
            client=self.cave,
            root_id=root_id,
            synapses=synapses,
            restore_graph=restore_graph,
            restore_properties=restore_properties,
            synapse_reference_tables=synapse_reference_tables,
            skeleton_version=skeleton_version,
        )
        if transform == "rigid":
            self.space.transform_nm.apply_meshwork_vertices(nrn, inplace=True)
            if synapses:
                space_cols = [
                    x for x in nrn.anno.pre_syn.df.columns if "pt_position" in x
                ]
                anno_dict = {"pre_syn": space_cols, "post_syn": space_cols}
                self.space.transform_nm.apply_meshwork_annotations(
                    nrn, anno_dict, inplace=True
                )
        elif transform == "streamline":
            self.space.streamline_nm.apply_meshwork_vertices(nrn, inplace=True)
            if synapses:
                space_cols = [
                    x for x in nrn.anno.pre_syn.df.columns if "pt_position" in x
                ]
                anno_dict = {"pre_syn": space_cols, "post_syn": space_cols}
                self.space.streamline_nm.apply_meshwork_annotations(
                    nrn, anno_dict, inplace=True
                )
        return nrn

    @staticmethod
    def now() -> datetime.datetime:
        """
        Get the current time in UTC timezone.
        """
        return datetime.datetime.now(datetime.timezone.utc)

    def version_timestamp(self, version: Optional[int] = None) -> datetime.datetime:
        """
        Get the timestamp for a specific materialization version.

        Parameters
        ----------
        version : int, optional
            The materialization version to get the timestamp for, by default None (uses current version).

        Returns
        -------
        datetime.datetime
            The timestamp of the specified materialization version.
        """
        if version is None:
            version = self.cave.materialize.version
        return self.cave.materialize.get_version_timestamp(version)

    def latest_valid_timestamp(
        self,
        root_ids: list[int],
    ) -> npt.NDArray:
        """
        Get the latest valid timestamps for a list of root IDs.
        If the root ID is out of date, it will return the last timestamp at which it was valid and could be used in queries.
        If the root ID is up to date, it will return the current timestamp at the request time, which is still ensured to be valid.

        Parameters
        ----------
        root_ids : list[int]
            The list of root IDs to get the latest valid timestamps for.

        Returns
        -------
        npt.NDArray
            The latest valid timestamps for the specified root IDs.
        """
        return self.cave.chunkedgraph.get_root_timestamps(root_ids, latest=True)

    def neuroglancer_url(
        self,
        target_url: Optional[str] = None,
        clipboard=False,
        shorten=False,
    ) -> str:
        """
        Get the Neuroglancer URL for the current datastack and version.

        Parameters
        ----------
        target_url : str, optional
            The base URL for Neuroglancer, by default None (uses default server address).

        Returns
        -------
        str
            The Neuroglancer URL.
        """
        vs = sb.ViewerState(client=self.cave).add_layers_from_client()
        if clipboard:
            return vs.to_clipboard(
                target_url=target_url,
                shorten=shorten,
            )
        else:
            return vs.to_url(
                target_url=target_url,
                shorten=shorten,
            )

    def __repr__(self) -> str:
        return f"DatasetClient(datastack_name={self.datastack_name}, version={self.cave.materialize.version})"

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Necessary for IPython to detect _repr_html_ for subclasses."""
        return {"text/html": self.__repr_html__()}, {}

    def __repr_html__(self) -> str:
        neuroglancer_url = self.neuroglancer_url()
        html_str = f"<html><body><a href='{neuroglancer_url}'>{self.__repr__()}</a></body></html>"
        return html_str
