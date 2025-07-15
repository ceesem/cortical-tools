import datetime
from collections import namedtuple
from typing import Optional

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from standard_transform.datasets import minnie_ds as microns_transform

from .utils import QueryTuple, current_timestamp

__all__ = [
    "client",
    "microns_transform",
    "current_timestamp",
    "queries",
    "cell_id_to_root_id",
    "root_id_to_cell_id",
]

DATASTACK_NAME = "minnie65_phase3_v1"
SERVER_ADDRESS = "https://global.daf-apis.com"

client = CAVEclient(
    datastack_name=DATASTACK_NAME,
    server_address=SERVER_ADDRESS,
)
queries = QueryTuple(
    tables=client.materialize.tables,
    views=client.materialize.views,
)


def cell_id_to_root_id(
    cell_ids: list[int],
    client: Optional[CAVEclient] = None,
    timestamp: Optional[datetime.datetime] = None,
    materialization_version: Optional[int] = None,
    filter_empty: bool = True,
) -> np.ndarray:
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
            datastack_name=DATASTACK_NAME,
            server_address=SERVER_ADDRESS,
        )
    view_name = "nucleus_detection_lookup_v1"
    nuc_df = (
        client.materialize.views[view_name](
            id=cell_ids,
        )
        .query(split_positions=True, materialization_version=materialization_version)
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
                lookup_df_alts.append(
                    lookup_df_alt[["id", "pt_root_id"]].set_index("pt_root_id")
                )
        if len(lookup_df_alts) > 0:
            lookup_df_alt_concat = pd.concat(lookup_df_alts)[
                ["id", "pt_root_id"]
            ].set_index("pt_root_id")
        else:
            lookup_df_alt_concat = []

        lookup_df = pd.concat([lookup_df_main, lookup_df_alt_concat])
    else:
        lookup_df = lookup_df_main
    return lookup_df


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
            datastack_name=DATASTACK_NAME,
            server_address=SERVER_ADDRESS,
        )

    root_ids = np.unique(root_ids)
    all_cell_df = pd.DataFrame(
        index=root_ids,
        data={"cell_id": -1, "done": False},
    )
    all_cell_df["cell_id"] = all_cell_df["cell_id"].astype(int)
    earliest_timestamp = client.chunkedgraph.get_root_timestamps(root_ids, latest=False)
    latest_timestamp = client.chunkedgraph.get_root_timestamps(root_ids, latest=True)
    all_cell_df["ts0"] = earliest_timestamp
    all_cell_df["ts1"] = latest_timestamp

    while not np.all(all_cell_df["done"].values):
        ts = all_cell_df.query("done == False").ts1.iloc[0]
        qry_idx = all_cell_df[(all_cell_df.ts0 < ts) & (all_cell_df.ts1 >= ts)].index

        lookup_df_main = client.materialize.tables.nucleus_detection_v0(
            pt_root_id=qry_idx.values
        ).live_query(timestamp=ts, split_positions=True)
        lookup_df_main = lookup_df_main[["id", "pt_root_id"]].set_index("pt_root_id")

        if len(lookup_df_main) < len(qry_idx):
            lookup_df_alt = client.materialize.tables.nucleus_alternative_points(
                pt_ref_root_id=qry_idx.values,
            ).live_query(timestamp=ts, split_positions=True)
            if len(lookup_df_alt) == 0:
                lookup_df_alt = lookup_df_alt[["id", "pt_root_id"]].set_index(
                    "pt_root_id"
                )
                lookup_df = lookup_df_main
            else:
                lookup_df = pd.concat([lookup_df_main, lookup_df_alt])
        else:
            lookup_df = lookup_df_main

        # Update the pt root ids of found cells, but the done status of all queried cells
        all_cell_df.loc[lookup_df.index, "cell_id"] = lookup_df["id"].astype(int)
        all_cell_df.loc[qry_idx, "done"] = True

    if filter_empty:
        all_cell_df = all_cell_df.query("cell_id != -1")
    return all_cell_df["cell_id"].astype(int).loc[root_ids]
