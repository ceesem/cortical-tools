import datetime

import numpy as np
import pandas as pd
import pytest

from cortical_tools import load_client

DATASTACK_LOOKUP = {
    "v1dd_client": "v1dd",
    "v1dd_public_client": "v1dd_public",
    "microns_prod_client": "minnie65_phase3_v1",
    "microns_public_client": "minnie65_public",
}


@pytest.mark.parametrize(
    "client_name",
    [
        "v1dd_client",
        "v1dd_public_client",
        "microns_prod_client",
        "microns_public_client",
    ],
)
def test_client_datastack(client_name, request):
    client = request.getfixturevalue(client_name)
    assert client.datastack_name == DATASTACK_LOOKUP.get(client_name)
    assert client.datastack_name == client.cave.datastack_name
    assert client.server_address == client.cave.server_address
    assert isinstance(client.now(), datetime.datetime)
    assert client.version_timestamp() == client.cave.materialize.get_timestamp(
        client.version
    )
    assert client.version == client.cave.materialize.version
    assert (
        isinstance(client.neuroglancer_url(), str) and client.neuroglancer_url() != ""
    )


@pytest.mark.parametrize(
    "client_name",
    [
        "v1dd_client",
        "v1dd_public_client",
        "microns_prod_client",
        "microns_public_client",
    ],
)
def test_client_basic_tables(client_name, request):
    client = request.getfixturevalue(client_name)
    assert len(client.tables) > 0


@pytest.mark.parametrize(
    "client_name",
    [
        "v1dd",
        "v1dd_public",
        "microns_prod",
        "microns_public",
    ],
)
def test_client_dynamic_load(client_name):
    client = load_client(client_name)
    assert client.datastack_name == DATASTACK_LOOKUP.get(f"{client_name}_client")


@pytest.mark.parametrize(
    "client_name",
    [
        "v1dd_client",
        "v1dd_public_client",
        "microns_prod_client",
        "microns_public_client",
    ],
)
def test_client_query_and_cell_lookup(client_name, request):
    client = request.getfixturevalue(client_name)
    lookup_table = client._root_id_lookup_main_table
    df = client.tables[lookup_table].get_all()
    df.drop_duplicates("pt_root_id", inplace=True, keep=False)
    sample_df = df.query("pt_root_id != 0").sample(1)

    root_id = int(sample_df["pt_root_id"].values[0])
    cell_id = int(sample_df["id"].values[0])

    looked_up_root_id = client.cell_id_to_root_id(cell_id)
    assert np.all(np.isin(root_id, np.array(looked_up_root_id)))

    looked_up_cell_id = client.root_id_to_cell_id(root_id)
    assert np.all(np.isin(cell_id, np.array(looked_up_cell_id)))


@pytest.mark.parametrize("transform", [None, "rigid", "streamline"])
@pytest.mark.parametrize(
    "client_name",
    [
        "v1dd_client",
        "v1dd_public_client",
        "microns_prod_client",
        "microns_public_client",
    ],
)
def test_client_skeleton(client_name, transform, request):
    client = request.getfixturevalue(client_name)
    lookup_table = client._root_id_lookup_main_table
    df = client.tables[lookup_table].get_all()
    df.drop_duplicates("pt_root_id", inplace=True, keep=False)

    good_sample = False
    while not good_sample:
        sample_df = df.query("pt_root_id != 0").sample(1)
        root_id = int(sample_df["pt_root_id"].values[0])

        l2_ids = client.get_l2_ids(root_id)
        assert len(l2_ids) > 0
        if len(l2_ids) > 100:
            good_sample = True

    nrn = client.get_skeleton(root_id, transform=transform)
    assert len(nrn.skeleton.vertices) > 0

    assert len(l2_ids) == len(nrn.mesh.vertices)
