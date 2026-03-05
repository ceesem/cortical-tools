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


ALL_CLIENTS = [
    "v1dd_client",
    "v1dd_public_client",
    "microns_prod_client",
    "microns_public_client",
]


@pytest.mark.parametrize("client_name", ALL_CLIENTS)
def test_client_chunkedgraph_wrappers(client_name, request):
    client = request.getfixturevalue(client_name)
    lookup_table = client._root_id_lookup_main_table
    df = client.tables[lookup_table].get_all()
    df.drop_duplicates("pt_root_id", inplace=True, keep=False)
    sample_df = df.query("pt_root_id != 0").sample(1)

    pt_root_id = int(sample_df["pt_root_id"].values[0])
    pt_supervoxel_id = int(sample_df["pt_supervoxel_id"].values[0])

    roots = client.get_roots([pt_supervoxel_id])
    assert pt_root_id in roots

    is_latest = client.is_latest_roots([pt_root_id])
    assert len(is_latest) == 1
    assert isinstance(is_latest[0], (bool, np.bool_))

    latest_roots = client.get_latest_roots(pt_root_id)
    assert isinstance(latest_roots, np.ndarray)
    assert len(latest_roots) > 0
    assert pt_root_id in latest_roots

    suggested = client.suggest_latest_roots(pt_root_id)
    assert isinstance(suggested, (int, np.integer, np.ndarray))


@pytest.mark.parametrize("client_name", ALL_CLIENTS)
def test_client_most_recent_materialization_version(client_name, request):
    client = request.getfixturevalue(client_name)
    assert (
        client.most_recent_materialization_version()
        == client.cave.materialize.most_recent_version()
    )


@pytest.mark.parametrize("client_name", ALL_CLIENTS)
def test_client_l2data(client_name, request):
    client = request.getfixturevalue(client_name)
    lookup_table = client._root_id_lookup_main_table
    df = client.tables[lookup_table].get_all()
    df.drop_duplicates("pt_root_id", inplace=True, keep=False)

    good_sample = False
    while not good_sample:
        sample_df = df.query("pt_root_id != 0").sample(1)
        root_id = int(sample_df["pt_root_id"].values[0])
        l2_ids = client.get_l2_ids(root_id)
        if len(l2_ids) > 0:
            good_sample = True

    result_df = client.get_l2data(l2_ids[:5])
    assert isinstance(result_df, pd.DataFrame)

    result_dict = client.get_l2data(l2_ids[:5], as_dataframe=False)
    assert isinstance(result_dict, dict)


@pytest.mark.parametrize("client_name", ALL_CLIENTS)
def test_client_skeletons_exist(client_name, request):
    client = request.getfixturevalue(client_name)
    lookup_table = client._root_id_lookup_main_table
    df = client.tables[lookup_table].get_all()
    df.drop_duplicates("pt_root_id", inplace=True, keep=False)
    sample_df = df.query("pt_root_id != 0").sample(2)

    root_ids = [int(v) for v in sample_df["pt_root_id"].values]
    result = client.skeletons_exist(root_ids=root_ids)
    assert isinstance(result, (bool, dict))


@pytest.mark.parametrize("client_name", ALL_CLIENTS)
def test_client_info_wrappers(client_name, request):
    client = request.getfixturevalue(client_name)

    info = client.get_datastack_info()
    assert isinstance(info, dict)
    assert "aligned_volume" in info

    img_src = client.image_source()
    assert isinstance(img_src, str) and img_src != ""

    seg_src = client.segmentation_source()
    assert isinstance(seg_src, str) and seg_src != ""

    resolution = client.viewer_resolution()
    assert isinstance(resolution, np.ndarray)
    assert resolution.shape == (3,)


@pytest.mark.parametrize("client_name", ALL_CLIENTS)
def test_client_state_roundtrip(client_name, request):
    client = request.getfixturevalue(client_name)
    state_id = client.upload_state_json({"test": True})
    assert isinstance(state_id, int)
    state = client.get_state_json(state_id)
    assert state == {"test": True}
