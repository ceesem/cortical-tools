import datetime
import logging
import warnings
from copy import copy
from itertools import combinations
from typing import TYPE_CHECKING, Optional, Self

warnings.filterwarnings(
    "ignore", message=".*Using `tqdm.autonotebook.tqdm` in notebook mode.*"
)


import fastremap
import gpytoolbox as gyp
import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse, spatial
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm_joblib import ParallelPbar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils import suppress_output

if TYPE_CHECKING:
    import numpy.typing as npt

    from .common import CAVEclientFull

__all__ = ["VertexAssigner"]


def get_lvl2_points(
    l2ids,
    caveclient,
) -> npt.NDArray:
    data = caveclient.l2cache.get_l2data(l2ids, attributes=["rep_coord_nm"])
    df = pd.DataFrame(
        {
            "lvl2_id": [int(x) for x in data.keys()],
            "pt_x": [x["rep_coord_nm"][0] for x in data.values()],
            "pt_y": [x["rep_coord_nm"][1] for x in data.values()],
            "pt_z": [x["rep_coord_nm"][2] for x in data.values()],
        }
    ).set_index("lvl2_id")

    return df.loc[l2ids][["pt_x", "pt_y", "pt_z"]].values


def bbox_mask(
    row,
    vertices,
    inclusive=True,
):
    """Create a mask for vertices within a bounding box defined by a row of chunk_df_solo

    Parameters
    ----------
    row : pd.Series
        A row from chunk_df_solo containing bounding box coordinates
    vertices : npt.NDArray
        Array of vertex positions

    Returns
    -------
    npt.NDArray
        Boolean mask indicating which vertices are within the bounding box
    """
    if inclusive:
        return (
            (vertices[:, 0] >= row["bbox_start_x"])
            & (vertices[:, 0] <= row["bbox_end_x"])
            & (vertices[:, 1] >= row["bbox_start_y"])
            & (vertices[:, 1] <= row["bbox_end_y"])
            & (vertices[:, 2] >= row["bbox_start_z"])
            & (vertices[:, 2] <= row["bbox_end_z"])
        )
    else:
        return (
            (vertices[:, 0] >= row["bbox_start_x"])
            & (vertices[:, 0] < row["bbox_end_x"])
            & (vertices[:, 1] >= row["bbox_start_y"])
            & (vertices[:, 1] < row["bbox_end_y"])
            & (vertices[:, 2] >= row["bbox_start_z"])
            & (vertices[:, 2] < row["bbox_end_z"])
        )


def chunk_to_nm(xyz_ch, cv):
    """Map a chunk location to Euclidean space

    Parameters
    ----------
    xyz_ch : array-like
        Nx3 array of chunk indices
    cv : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space
    voxel_resolution : list, optional
        Voxel resolution, by default [4, 4, 40]

    Returns
    -------
    np.array
        Nx3 array of spatial points
    """
    base_location = cv.meta.voxel_offset(0) * cv.mip_resolution(0)
    x_vox = np.atleast_2d(xyz_ch) * cv.meta.graph_chunk_size * cv.mip_resolution(0)
    return base_location + x_vox


def component_submesh(
    within_component_mask,
    vertices,
    faces,
):
    """Create a submesh for the specific component of the mesh"""
    face_touch_component = np.any(within_component_mask[faces], axis=1)
    component_faces = faces[face_touch_component]
    if len(component_faces) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 3), dtype=int)
    newV, newF = gyp.remove_unreferenced(vertices, component_faces)
    return newV, newF


def create_component_dict(chunk_rows, vertices, faces) -> list:
    # Reduce to an edge-inclusive collection of vertices and faces.
    mask_all = bbox_mask(chunk_rows.iloc[0], vertices, inclusive=True)
    vertices_chunk = vertices[mask_all]
    faces_filter = faces[np.all(mask_all[faces], axis=1)]
    relabel = {v: k for k, v in enumerate(np.flatnonzero(mask_all))}
    faces_chunk = fastremap.remap(
        faces_filter,
        relabel,
    )

    # Now go to the subset of vertices and faces purely within the chunk
    mask_in = bbox_mask(chunk_rows.iloc[0], vertices_chunk, inclusive=False)
    if not np.any(mask_in):
        return []
    if np.all(mask_in):
        faces_not_touching_edge = faces_chunk
    else:
        faces_not_touching_edge = faces_chunk[np.all(mask_in[faces_chunk], axis=1)]

    # Make sure isolated vertices are included in the components, but only if interior
    face_identity = np.array([[ii, ii, ii] for ii in range(mask_in.shape[0])])
    vertex_cc = gyp.connected_components(
        np.vstack(
            [np.atleast_2d(faces_not_touching_edge).reshape(-1, 3), face_identity]
        )
    )
    vertex_cc[~mask_in] = -1

    # Now for each component, find the faces associated with its true vertices plus any faces that are only on the boundary of the chunk
    components = []
    assigned_vertices = np.full(mask_all.shape[0], False, dtype=bool)
    comp_id = 0
    for ii in np.unique(vertex_cc[mask_in]):
        comp_mask = np.full(mask_all.shape, False, dtype=bool)
        comp_mask[mask_all] = vertex_cc == ii
        comp_verts, comp_faces = component_submesh(
            vertex_cc == ii, vertices_chunk, faces_chunk
        )
        assigned_vertices[comp_mask] = True
        if comp_faces.shape[0] == 0:
            continue
        components.append(
            {
                "component_id": comp_id,
                "vertices": comp_verts,
                "faces": comp_faces,
                "mask": comp_mask,
                "vertices_in": vertices[comp_mask],
            }
        )
        comp_id += 1
    return components


class VertexAssigner:
    def __init__(
        self,
        root_id: int,
        caveclient: Optional["CAVEclientFull"] = None,
        vertices: Optional[npt.NDArray] = None,
        faces: Optional[npt.NDArray] = None,
        lvl2_ids: Optional[npt.NDArray] = None,
        lvl2_pts: Optional[npt.NDArray] = None,
        lru_cache: Optional[int] = 10 * 1024,
    ):
        self.caveclient = caveclient
        self.cv = self.caveclient.info.segmentation_cloudvolume()
        if lru_cache is not None:
            self.cv.image.lru.resize(lru_cache)

        self._root_id = root_id
        if vertices is not None and faces is not None:
            self._vertices = vertices
            self._faces = faces
        else:
            self._vertices = None
            self._faces = None
        self._timestamp = None
        self._chunk_df_solo = None
        self._chunk_df_multi = None
        self._mesh_label = None
        self._lvl2_ids = lvl2_ids
        self._lvl2_pts = lvl2_pts
        self._setup_root_id()

    def _setup_root_id(self) -> Self:
        """Set the root ID for the mesh"""
        if self._vertices is None or self._faces is None:
            logger.info("Fetching mesh data for root ID: %d", self._root_id)
            self._vertices, self._faces = self.get_mesh_data(self._root_id)
        self._timestamp = self.root_id_timestamp(self._root_id)
        self._chunk_df_solo, self._chunk_df_multi = self.get_chunk_dataframes(
            self._root_id, self._lvl2_ids, self._lvl2_pts
        )
        return self

    @property
    def root_id(self) -> int:
        """Get the root ID for the mesh"""
        if self._root_id is None:
            raise ValueError("Root ID must be set before accessing it.")
        return self._root_id

    @property
    def vertices(self) -> npt.NDArray:
        """Get the vertices of the mesh"""
        if self._vertices is None:
            raise ValueError("Vertices must be set before accessing them.")
        return self._vertices

    @property
    def faces(self) -> npt.NDArray:
        """Get the faces of the mesh"""
        if self._faces is None:
            raise ValueError("Faces must be set before accessing them.")
        return self._faces

    @property
    def timestamp(self) -> datetime.datetime:
        """Get the timestamp for the root ID"""
        if self._timestamp is None:
            raise ValueError("Timestamp must be set before accessing it.")
        return self._timestamp

    @property
    def chunk_df_solo(self) -> pd.DataFrame:
        """Get the chunk dataframe for solo chunks"""
        if self._chunk_df_solo is None:
            raise ValueError("Chunk dataframe must be set before accessing it.")
        return self._chunk_df_solo

    @property
    def chunk_df_multi(self) -> pd.DataFrame:
        """Get the chunk dataframe for multi-component chunks"""
        if self._chunk_df_multi is None:
            raise ValueError("Chunk dataframe must be set before accessing it.")
        return self._chunk_df_multi

    @property
    def chunk_df(self) -> pd.DataFrame:
        """Get the chunk dataframe for the mesh"""
        if self._chunk_df_solo is None or self._chunk_df_multi is None:
            raise ValueError("Chunk dataframes must be set before accessing them.")
        return pd.concat([self._chunk_df_solo, self._chunk_df_multi]).sort_index()

    @property
    def lvl2_ids(self) -> npt.NDArray:
        return self.chunk_df["l2id"].values

    @property
    def mesh_label_index(self) -> npt.NDArray:
        """Get the mesh label index into the lvl2 ids for the vertices"""
        if self._mesh_label is None:
            raise ValueError(
                "Mesh label must be computed with 'get_mesh_label' before accessing it."
            )
        return self._mesh_label

    def chunk_to_nm(self, xyz_ch: npt.NDArray) -> npt.NDArray:
        """Map a chunk location to Euclidean space

        Parameters
        ----------
        xyz_ch : array-like
            Nx3 array of chunk indices

        Returns
        -------
        np.array
            Nx3 array of spatial points
        """
        return chunk_to_nm(xyz_ch, self.cv)

    @property
    def chunk_dims(self) -> npt.NDArray:
        """Gets the size of a chunk in euclidean space

        Parameters
        ----------
        cv : cloudvolume.CloudVolume
            Chunkedgraph-targeted cloudvolume object

        Returns
        -------
        np.array
            3-element box dimensions of a chunk in nanometers.
        """
        dims = chunk_to_nm([1, 1, 1], self.cv) - chunk_to_nm([0, 0, 0], self.cv)
        return np.squeeze(dims)

    @property
    def draco_size(self) -> int:
        """Get the size of a draco grid in nanometers"""
        return self.cv.meta.get_draco_grid_size(0)

    def adjust_for_draco(
        self,
        vals: npt.NDArray,
    ) -> npt.NDArray:
        "Adjust grid locations to align with the discrete draco grid"
        return self.draco_size * np.floor(vals / self.draco_size)

    def make_chunk_bbox(self, l2ids, adjust_draco=True):
        chunk_numbers = [
            int(self.cv.meta.decode_chunk_position_number(l)) for l in l2ids
        ]
        chunk_grid = np.array(
            [np.array(self.cv.meta.decode_chunk_position(l)) for l in l2ids]
        )
        chunk_start = self.chunk_to_nm(chunk_grid)
        chunk_end = chunk_start + self.chunk_dims

        if adjust_draco:
            chunk_start = self.adjust_for_draco(chunk_start)
            chunk_end = self.adjust_for_draco(chunk_end)

        df = pd.DataFrame(
            {
                "l2id": l2ids.astype(int),
                "chunk_x": chunk_grid[:, 0],
                "chunk_y": chunk_grid[:, 1],
                "chunk_z": chunk_grid[:, 2],
                "bbox_start_x": chunk_start[:, 0],
                "bbox_start_y": chunk_start[:, 1],
                "bbox_start_z": chunk_start[:, 2],
                "bbox_end_x": chunk_end[:, 0],
                "bbox_end_y": chunk_end[:, 1],
                "bbox_end_z": chunk_end[:, 2],
                "chunk_number": chunk_numbers,
            }
        )
        return df

    def chunk_dataframe(self, l2ids: npt.NDArray, points: npt.NDArray) -> pd.DataFrame:
        """Create a dataframe of chunk bounding boxes for a neuron

        Parameters
        ----------
        l2ids : array-like
            List of level 2 IDs
        points : pd.DataFrame
            DataFrame containing point coordinates
        cv : cloudvolume.CloudVolume
            CloudVolume object associated with the chunked space

        Returns
        -------
        pd.DataFrame
            DataFrame containing bounding boxes for each chunk in the neuron
        """
        df = self.make_chunk_bbox(l2ids)
        pt_df = pd.DataFrame(
            {
                "l2id": l2ids.astype(int),
                "pt_x": points[:, 0],
                "pt_y": points[:, 1],
                "pt_z": points[:, 2],
            }
        )
        return df.merge(
            pt_df,
            on="l2id",
            how="left",
        )

    def assign_points_to_components(
        self,
        chunk_rows,
        vertices,
        faces,
        ts: Optional[bool] = None,
        cloudvolume_fallback: bool = False,
        max_distance: float = 500,
        ratio_better: float = 0.33,
        coarse: bool = False,
    ):
        """Assign representative points to components in a chunk mesh.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the index of the representative point in the chunk_rows dataframe and the component ID.

        dict
            Dictionary mapping component IDs to masks for the vertices in the global mesh.
        """
        pts = np.array(chunk_rows[["pt_x", "pt_y", "pt_z"]].values, dtype=float)
        components = create_component_dict(chunk_rows, vertices, faces)
        if len(components) == 0:
            return (
                pd.DataFrame(
                    {
                        "representative_pt": [],
                        "graph_comp": [],
                    }
                ),
                {},
            )
        wn_results = []
        for comp in components:
            wn_results.append(
                gyp.fast_winding_number(pts, comp["vertices"], comp["faces"])
            )
        wn_results = np.array(wn_results).T

        pt_assign, mesh_assign = linear_sum_assignment(
            np.array(wn_results) / np.max(wn_results), maximize=True
        )

        # If there are more components than points, don't assign components to the lower-scoring ones
        if len(pts) < len(components):
            mesh_assign = mesh_assign[: len(pt_assign)]
        if len(pts) > len(components):
            pt_assign = pt_assign[: len(mesh_assign)]
        result_df = pd.DataFrame(
            {
                "representative_pt": pt_assign.astype(int),
                "graph_comp": mesh_assign.astype(int),
            }
        )
        if len(result_df) == 0 and coarse:
            return None, None, None

        if not coarse:
            for comp in components:
                # If you have not already assigned a point to this component, use the slower cloudvolume lookup

                if comp["component_id"] not in result_df["graph_comp"].values:
                    self.representative_point_via_proximity(
                        components=components,
                        result_df=result_df,
                        max_distance=max_distance,
                        ratio_better=ratio_better,
                    )
                    if len(result_df) < len(components) and cloudvolume_fallback:
                        point_to_component = self.representative_point_via_lookup(
                            chunk_rows=chunk_rows,
                            comp=comp,
                            timestamp=ts,
                        )
                        if point_to_component == -1:
                            continue
                        result_df.loc[result_df.index[-1] + 1] = {
                            "representative_pt": point_to_component,
                            "graph_comp": comp["component_id"],
                        }

        comp_mask_dict = {}
        for comp in components:
            if comp["component_id"] in result_df["graph_comp"].values:
                comp_mask_dict[comp["component_id"]] = comp["mask"]

        return result_df, comp_mask_dict

    def representative_point_via_proximity(
        self,
        components: dict,
        result_df: pd.DataFrame,
        max_distance: float = 250,
        ratio_better: float = 0.25,
    ):
        """For unassigned components, find the representative closest point on each assigned component. Do the assignment if it is a clear winner"""

        assigned_components = result_df["graph_comp"].unique()

        first_comps = []
        second_comps = []
        first_assigned = []
        second_assigned = []
        ds = []
        kdtrees = [spatial.KDTree(comp["vertices"]) for comp in components]
        for comp_a, comp_b in combinations(components, 2):
            first_comps.append(comp_a["component_id"])
            second_comps.append(comp_b["component_id"])
            first_assigned.append(comp_a["component_id"] in assigned_components)
            second_assigned.append(comp_b["component_id"] in assigned_components)
            if not (
                comp_a["component_id"] in assigned_components
                and comp_b["component_id"] in assigned_components
            ):
                comp_ds = np.array(
                    list(
                        kdtrees[comp_a["component_id"]]
                        .sparse_distance_matrix(
                            kdtrees[comp_b["component_id"]],
                            max_distance=max_distance / ratio_better,
                            output_type="dok_matrix",
                        )
                        .values()
                    )
                )
                if len(comp_ds) > 0:
                    ds.append(np.min(comp_ds))
                else:
                    ds.append(np.inf)
            else:
                ds.append(np.inf)
        distance_graph = pd.DataFrame(
            {
                "first_comp": first_comps,
                "second_comp": second_comps,
                "first_assigned": first_assigned,
                "second_assigned": second_assigned,
                "distance": ds,
            }
        )
        distance_graph = distance_graph[
            distance_graph["distance"] < max_distance
        ].reset_index()
        distance_graph["evaluated"] = False

        # Ignore distances that are too large (and note that we set evaluated pairs to infinity)
        while not np.all(distance_graph["evaluated"]):
            pairs_to_consider = distance_graph.query(
                "evaluated == False and first_assigned != second_assigned"
            ).sort_values("distance")
            if len(pairs_to_consider) == 0:
                break
            for gph_idx, row in pairs_to_consider.iterrows():
                distance_graph.loc[gph_idx, "evaluated"] = True
                if row["first_assigned"]:
                    assigned_comp = row["first_comp"]
                    unassigned_comp = row["second_comp"]
                else:
                    assigned_comp = row["second_comp"]
                    unassigned_comp = row["first_comp"]
                ds_edge = (
                    pairs_to_consider.drop(index=gph_idx)
                    .query(
                        "first_comp == @unassigned_comp or second_comp == @unassigned_comp and evaluated == False and first_assigned!=second_assigned"
                    )["distance"]
                    .values
                )
                do_assign = False
                if len(ds_edge) == 0:
                    do_assign = True
                elif row["distance"] < ratio_better * np.min(ds_edge):
                    do_assign = True
                if do_assign:
                    best_pt = result_df.query("graph_comp == @assigned_comp")[
                        "representative_pt"
                    ].values[0]
                    result_df.loc[result_df.index[-1] + 1] = {
                        "representative_pt": best_pt,
                        "graph_comp": unassigned_comp,
                    }
                    distance_graph.loc[
                        distance_graph["first_comp"] == unassigned_comp,
                        "first_assigned",
                    ] = True
                    distance_graph.loc[
                        distance_graph["second_comp"] == unassigned_comp,
                        "second_assigned",
                    ] = True

        return result_df

    def find_closest_assigned_component(
        self,
        comp: dict,
        vert_assigned: dict,
        max_distance: float,  # Maximum distance to consider for assignment
        ratio_better: float,  # Ratio of distance to the best component to the second best to consider it a clear winner. Should be less than one.
    ):
        if len(vert_assigned) == 1:
            return list(vert_assigned.keys())[0]
        ds = np.array(
            [np.min(cdist(comp["vertices"], v)) for v in vert_assigned.values()]
        )
        # ds[ds == 0] = np.inf  # Ignore zero distances, since they would be attached
        dist_sort = np.argsort(ds)
        # if the closest component is significantly closer than the second closest and not above some threshold, assign it
        if (
            ds[dist_sort[0]] < ratio_better * ds[dist_sort[1]]
            and ds[dist_sort[0]] < max_distance
        ):
            return list(vert_assigned.keys())[dist_sort[0]]
        else:
            return -1

    def get_mesh_l2id_from_lookup(
        self,
        comp: dict,
        timestamp: datetime.datetime,
        point_counts: Optional[list[int]] = None,
        potential_l2ids: npt.NDArray = None,
    ):
        comp_bbox = np.vstack(
            [
                np.min(comp["vertices"], axis=0) - 5 * self.draco_size,
                5 * self.draco_size + np.max(comp["vertices"], axis=0),
            ]
        )
        not_enough_points = True
        point_counts = [400, 1000] if point_counts is None else point_counts

        while not_enough_points:
            if len(point_counts) == 0:
                return -1
            N = point_counts.pop(0)
            random_points = np.random.uniform(
                low=comp_bbox[0], high=comp_bbox[1], size=(N, 3)
            ).astype(int)
            with suppress_output():
                pt_lookup = np.array(
                    list(
                        self.cv.scattered_points(
                            random_points,
                            mip=0,
                            coord_resolution=[1, 1, 1],
                            agglomerate=True,
                            stop_layer=2,
                            timestamp=timestamp,
                        ).values()
                    )
                )

            if potential_l2ids is None:
                potential_l2ids = np.unique(pt_lookup)
            point_in_root = np.isin(pt_lookup, potential_l2ids)
            if np.sum(point_in_root) >= 1:
                not_enough_points = False
            elif point_in_root.sum() == 0:
                print("No points found in the root. Trying again with more points.")
                continue
            l2ids, counts = np.unique(pt_lookup[point_in_root], return_counts=True)
            return l2ids[np.argmax(counts)]

    def representative_point_via_lookup(
        self,
        chunk_rows,
        comp,
        timestamp,
        point_counts=None,
    ):
        l2ids = chunk_rows["l2id"].values
        l2id = self.get_mesh_l2id_from_lookup(
            comp,
            point_counts=copy(point_counts),
            potential_l2ids=l2ids,
            timestamp=timestamp,
        )
        if l2id == -1:
            return -1
        else:
            return int(np.flatnonzero(l2ids == l2id)[0])

    def process_multicomponent_chunk(
        self,
        chunk_rows: pd.DataFrame,
        vertices: npt.NDArray,
        faces: npt.NDArray,
        ts: datetime.datetime,
        cloudvolume_fallback: bool = False,
        max_distance: float = 500,
        ratio_better: float = 0.33,
        coarse: bool = False,
    ) -> list:
        """Process a single mesh chunk

            Parameters
            ----------
            chunk_rows : pd.DataFrame
                DataFrame containing chunk bounding box information and vertex positions
        vertices : npt.NDArray
                Array of vertex positions for the complete mesh
        faces : npt.NDArray
                Array of face indices for the complete mesh
            ts: datetime.datetime
                Timestamp for the root id

            Returns
            -------
            tuple
                A tuple containing two arrays, both with one entry for every vertex contained in the chunk bounding box:
                - `mind`: Indices of mesh vertices in the chunk
                - `l2id_index`: Indices of the representative points in the chunk as defined by the chunk_rows DataFrame
        """

        # To get the right mesh faces for association, we need to include the vertices on chunk bounds even if we don't plan to assign values to them
        assignment_df, component_mask_dict = self.assign_points_to_components(
            chunk_rows,
            vertices,
            faces,
            ts,
            cloudvolume_fallback=cloudvolume_fallback,
            max_distance=max_distance,
            ratio_better=ratio_better,
            coarse=coarse,
        )
        if assignment_df is None:
            return []

        id_mapping = []
        for _, row in assignment_df.iterrows():
            id_mapping.append(
                {
                    "l2id_idx": int(chunk_rows.index[row["representative_pt"]]),
                    "vertex_mask": np.flatnonzero(
                        component_mask_dict[row["graph_comp"]]
                    ),
                }
            )
        return id_mapping

    def get_l2_components(
        self,
        root_id,
        caveclient=None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if caveclient is None:
            caveclient = self.caveclient
        l2ids = caveclient.chunkedgraph.get_leaves(root_id, stop_layer=2)
        rep_points = get_lvl2_points(l2ids, caveclient)
        return l2ids, rep_points

    def get_mesh_data(
        self,
        root_id,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        with suppress_output():
            mesh = self.cv.mesh.get(root_id, fuse=False).get(root_id)
        return mesh.vertices, mesh.faces

    def get_chunk_dataframes(
        self,
        caveclient: Optional["CAVEclientFull"] = None,
        lvl2_ids: Optional[npt.NDArray] = None,
        lvl2_pts: Optional[npt.NDArray] = None,
    ) -> pd.DataFrame:
        """Get chunk dataframe for a neuron

        Parameters
        ----------
        root_id : int
            Root ID for a neuron
        caveclient : Optional[CAVEclientFull], optional
            CAVE client, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame containing chunk bounding boxes and representative points
        """
        if self._root_id is None:
            raise ValueError(
                "Root ID must be set before processing multi-component chunks."
            )

        if caveclient is None:
            caveclient = self.caveclient
        if lvl2_ids is None or lvl2_pts is None:
            l2ids, rep_points = self.get_l2_components(self._root_id)
        else:
            l2ids = lvl2_ids
            rep_points = lvl2_pts
        df = self.chunk_dataframe(l2ids, rep_points)
        df_solo = df.drop_duplicates("chunk_number", keep=False)
        df_multi = df[df.duplicated("chunk_number", keep=False)]
        return df_solo, df_multi

    def process_chunk_dataframe_solo(
        self,
    ) -> list:
        if self._root_id is None:
            raise ValueError(
                "Root ID must be set before processing multi-component chunks."
            )

        id_mapping = []
        vertex_lists = Parallel(n_jobs=-1)(
            delayed(bbox_mask)(row, self._vertices)
            for _, row in self._chunk_df_solo.iterrows()
        )
        for idx, vert_mask in zip(self._chunk_df_solo.index, vertex_lists):
            id_mapping.append(
                {
                    "l2id_idx": int(idx),
                    "vertex_mask": np.flatnonzero(vert_mask),
                }
            )
        return id_mapping

    def root_id_timestamp(
        self,
        root_id: int,
    ):
        """Get the timestamp for a root ID

        Parameters
        ----------
        root_id : int
            Root ID for a neuron

        Returns
        -------
        datetime.datetime
            Timestamp for the root ID
        """
        return self.caveclient.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]

    def process_chunk_dataframe_multi(
        self,
        cloudvolume_fallback: bool = False,
        max_distance: float = 500,
        ratio_better: float = 0.33,
        coarse: bool = False,
        n_jobs: int = -1,
    ) -> list:
        if self._root_id is None:
            raise ValueError(
                "Root ID must be set before processing multi-component chunks."
            )
        chunk_groups = [
            chunk_rows for _, chunk_rows in self._chunk_df_multi.groupby("chunk_number")
        ]

        # Parallelize over chunks (process-based)
        id_mapping_results = ParallelPbar(desc="Processing complex chunks...")(
            n_jobs=n_jobs,
            prefer="processes",
        )(
            delayed(self.process_multicomponent_chunk)(
                chunk_rows,
                self._vertices,
                self._faces,
                self._timestamp,
                cloudvolume_fallback=cloudvolume_fallback,
                max_distance=max_distance,
                ratio_better=ratio_better,
                coarse=coarse,
            )
            for chunk_rows in chunk_groups
        )

        # Flatten to a single list[dict]
        id_mapping: list = []
        for result in id_mapping_results:
            if isinstance(result, list) and len(result) > 0:
                id_mapping.extend(result)
        return id_mapping

    def propagate_labels(
        self,
        hop_limit: int = 50,
    ):
        A = gyp.adjacency_matrix(
            self.faces,
        )

        labeled_inds = np.flatnonzero(self.mesh_label_index != -1)

        d_to, _, p2 = sparse.csgraph.dijkstra(
            A,
            indices=labeled_inds,
            limit=hop_limit,
            min_only=True,
            return_predecessors=True,
            unweighted=True,
        )

        unlabeled_inds = np.flatnonzero((self.mesh_label_index == -1) & ~np.isinf(d_to))
        self._mesh_label[unlabeled_inds] = self._mesh_label[p2[unlabeled_inds]]
        return self.mesh_label

    def compute_mesh_label(
        self,
        max_distance: float = 500,
        ratio_better: float = 0.5,
        cloudvolume_fallback: bool = False,
        hop_limit: Optional[int] = None,
        coarse: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Process the mesh.

        Returns
        -------
        np.ndarray
            Array of l2ids for each mesh label. Unassigned values have id 0.
        """
        if hop_limit is None:
            if coarse:
                hop_limit = 75
            else:
                hop_limit = 50

        logger.info("Processing simple chunks...")
        id_mapping_solo = self.process_chunk_dataframe_solo()
        logger.info("Processing complex chunks...")
        id_mapping_multi = self.process_chunk_dataframe_multi(
            cloudvolume_fallback=cloudvolume_fallback,
            max_distance=max_distance,
            ratio_better=ratio_better,
            coarse=coarse,
            n_jobs=n_jobs,
        )

        mesh_label = np.full(self.vertices.shape[0], -1, dtype=int)
        for row in id_mapping_solo:
            mesh_label[row["vertex_mask"]] = row["l2id_idx"]
        for row in id_mapping_multi:
            mesh_label[row["vertex_mask"]] = row["l2id_idx"]
        self._mesh_label = mesh_label
        if hop_limit > 0:
            self.propagate_labels(hop_limit=hop_limit)
        return self._lvl2_map()

    @property
    def mesh_label(self) -> np.ndarray:
        return self._lvl2_map()

    def _lvl2_map(self) -> np.ndarray:
        lvl2_map = np.full(self.vertices.shape[0], 0, dtype=int)
        lvl2_map[self.mesh_label_index != -1] = self.lvl2_ids[
            self.mesh_label_index[self.mesh_label_index != -1]
        ]
        return lvl2_map
