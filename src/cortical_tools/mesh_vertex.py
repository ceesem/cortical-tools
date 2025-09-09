import datetime
import gc
import logging
import os
import warnings
from copy import copy
from itertools import combinations
from typing import TYPE_CHECKING, Optional, Self

import psutil

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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils import suppress_output

if TYPE_CHECKING:
    import numpy.typing as npt

    from .common import CAVEclientFull

__all__ = ["VertexAssigner"]


def log_memory_usage(stage_name: str, logger=logger, level=logging.DEBUG):
    """Log current memory usage for debugging"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.log(level, f"Memory usage at {stage_name}: {memory_mb:.1f} MB")
    return memory_mb


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


def spatial_bbox_query(
    row,
    vertices,
    spatial_index,
    inclusive=True,
):
    """Efficient spatial query for vertices within a bounding box using KDTree

    Parameters
    ----------
    row : pd.Series
        A row from chunk_df_solo containing bounding box coordinates
    vertices : npt.NDArray
        Array of vertex positions (used for compatibility)
    spatial_index : scipy.spatial.cKDTree
        Pre-built spatial index of the vertices
    inclusive : bool
        Whether to include boundary vertices

    Returns
    -------
    npt.NDArray
        Boolean mask indicating which vertices are within the bounding box
    """
    # Extract bounding box coordinates
    min_bounds = np.array(
        [row["bbox_start_x"], row["bbox_start_y"], row["bbox_start_z"]]
    )
    max_bounds = np.array([row["bbox_end_x"], row["bbox_end_y"], row["bbox_end_z"]])

    # Use rectangular query approach optimized for chunk structure
    # Since chunks are roughly cubic, use the diagonal distance as a conservative bound
    center = (min_bounds + max_bounds) / 2
    chunk_dimensions = max_bounds - min_bounds

    # Use half the diagonal of the bounding box as radius for L2 norm query
    # This ensures we capture all vertices in the rectangular region
    diagonal_radius = np.linalg.norm(chunk_dimensions) / 2

    # Get candidate vertices within the bounding sphere (L2 norm is faster than L-inf for KDTree)
    candidate_indices = spatial_index.query_ball_point(center, diagonal_radius, p=2)

    # Create boolean mask
    mask = np.zeros(len(vertices), dtype=bool)

    if len(candidate_indices) > 0:
        # Filter candidates to exact bounding box
        candidate_vertices = vertices[candidate_indices]

        if inclusive:
            exact_mask = (
                (candidate_vertices[:, 0] >= min_bounds[0])
                & (candidate_vertices[:, 0] <= max_bounds[0])
                & (candidate_vertices[:, 1] >= min_bounds[1])
                & (candidate_vertices[:, 1] <= max_bounds[1])
                & (candidate_vertices[:, 2] >= min_bounds[2])
                & (candidate_vertices[:, 2] <= max_bounds[2])
            )
        else:
            exact_mask = (
                (candidate_vertices[:, 0] >= min_bounds[0])
                & (candidate_vertices[:, 0] < max_bounds[0])
                & (candidate_vertices[:, 1] >= min_bounds[1])
                & (candidate_vertices[:, 1] < max_bounds[1])
                & (candidate_vertices[:, 2] >= min_bounds[2])
                & (candidate_vertices[:, 2] < max_bounds[2])
            )

        # Set mask for valid candidates
        valid_candidates = np.array(candidate_indices)[exact_mask]
        mask[valid_candidates] = True

    return mask


def vectorized_bbox_batch_query(
    chunk_rows,
    vertices,
    inclusive=True,
):
    """Vectorized batch processing of bounding box queries for multiple chunks

    This replaces individual spatial queries with a single vectorized operation,
    dramatically reducing function call overhead and improving cache locality.

    Parameters
    ----------
    chunk_rows : list of tuples
        List of (index, row) pairs from chunk dataframe
    vertices : npt.NDArray
        Array of vertex positions
    inclusive : bool
        Whether to include boundary vertices

    Returns
    -------
    list of npt.NDArray
        List of boolean masks, one for each chunk
    """
    if len(chunk_rows) == 0:
        return []

    n_chunks = len(chunk_rows)
    n_vertices = len(vertices)

    # Pre-allocate arrays for better memory efficiency
    min_bounds = np.empty((n_chunks, 3), dtype=vertices.dtype)
    max_bounds = np.empty((n_chunks, 3), dtype=vertices.dtype)

    # Extract bounding box data directly into pre-allocated arrays
    for i, (idx, row) in enumerate(chunk_rows):
        min_bounds[i, 0] = row["bbox_start_x"]
        min_bounds[i, 1] = row["bbox_start_y"]
        min_bounds[i, 2] = row["bbox_start_z"]
        max_bounds[i, 0] = row["bbox_end_x"]
        max_bounds[i, 1] = row["bbox_end_y"]
        max_bounds[i, 2] = row["bbox_end_z"]

    # Pre-allocate result list for better performance
    result_masks = [None] * n_chunks

    # Vectorized processing - optimized for memory access patterns
    for chunk_idx in range(n_chunks):
        # Direct array access is faster than slicing
        min_x, min_y, min_z = min_bounds[chunk_idx]
        max_x, max_y, max_z = max_bounds[chunk_idx]

        # Optimized vectorized comparison with short-circuit evaluation
        if inclusive:
            mask = (
                (vertices[:, 0] >= min_x)
                & (vertices[:, 0] <= max_x)
                & (vertices[:, 1] >= min_y)
                & (vertices[:, 1] <= max_y)
                & (vertices[:, 2] >= min_z)
                & (vertices[:, 2] <= max_z)
            )
        else:
            mask = (
                (vertices[:, 0] >= min_x)
                & (vertices[:, 0] < max_x)
                & (vertices[:, 1] >= min_y)
                & (vertices[:, 1] < max_y)
                & (vertices[:, 2] >= min_z)
                & (vertices[:, 2] < max_z)
            )

        result_masks[chunk_idx] = mask

    return result_masks


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
    # Reduce to an edge-inclusive collection of vertices and faces using vectorized approach
    first_row = chunk_rows.iloc[0]

    # Use vectorized bbox query instead of old bbox_mask
    mask_all = (
        (vertices[:, 0] >= first_row["bbox_start_x"])
        & (vertices[:, 0] <= first_row["bbox_end_x"])
        & (vertices[:, 1] >= first_row["bbox_start_y"])
        & (vertices[:, 1] <= first_row["bbox_end_y"])
        & (vertices[:, 2] >= first_row["bbox_start_z"])
        & (vertices[:, 2] <= first_row["bbox_end_z"])
    )

    vertices_chunk = vertices[mask_all]
    faces_filter = faces[np.all(mask_all[faces], axis=1)]
    relabel = {v: k for k, v in enumerate(np.flatnonzero(mask_all))}
    faces_chunk = fastremap.remap(
        faces_filter,
        relabel,
    )

    # Now go to the subset of vertices and faces purely within the chunk (non-inclusive)
    mask_in = (
        (vertices_chunk[:, 0] >= first_row["bbox_start_x"])
        & (vertices_chunk[:, 0] < first_row["bbox_end_x"])
        & (vertices_chunk[:, 1] >= first_row["bbox_start_y"])
        & (vertices_chunk[:, 1] < first_row["bbox_end_y"])
        & (vertices_chunk[:, 2] >= first_row["bbox_start_z"])
        & (vertices_chunk[:, 2] < first_row["bbox_end_z"])
    )
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
        log_memory_usage("start of _setup_root_id")

        if self._vertices is None or self._faces is None:
            logger.info("Fetching mesh data for root ID: %d", self._root_id)
            self._vertices, self._faces = self.get_mesh_data(self._root_id)
            log_memory_usage("after loading mesh data")
            logger.info(
                f"Loaded mesh with {len(self._vertices)} vertices and {len(self._faces)} faces"
            )

        self._timestamp = self.root_id_timestamp(self._root_id)
        log_memory_usage("after getting timestamp")

        self._chunk_df_solo, self._chunk_df_multi = self.get_chunk_dataframes(
            self._root_id, self._lvl2_ids, self._lvl2_pts
        )
        log_memory_usage("after getting chunk dataframes")
        logger.info(
            f"Created {len(self._chunk_df_solo)} solo chunks and {len(self._chunk_df_multi)} multi-component chunks"
        )

        # Vectorized batch processing doesn't need spatial index
        logger.info("Using vectorized batch processing (no spatial index needed)")

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
            np.array(wn_results) / (np.max(wn_results) + 1), maximize=True
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
            with suppress_output("urllib3"):
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
        with suppress_output("urllib3"):
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
        batch_size: int = 5000,
    ) -> list:
        if self._root_id is None:
            raise ValueError(
                "Root ID must be set before processing multi-component chunks."
            )

        log_memory_usage("start of process_chunk_dataframe_solo", level=logging.DEBUG)
        logger.info(
            f"Processing {len(self._chunk_df_solo)} solo chunks with vectorized batch queries"
        )

        # Track overall performance
        import time

        overall_start = time.time()

        # Process in batches to limit memory usage
        id_mapping = []
        chunk_rows = list(self._chunk_df_solo.iterrows())
        total_batches = (len(chunk_rows) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunk_rows))
            batch_rows = chunk_rows[start_idx:end_idx]

            logger.debug(
                f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_rows)} chunks)"
            )

            # Time the vectorized batch query performance
            import time

            batch_start = time.time()

            # Use vectorized batch processing - eliminates multiprocessing overhead
            vertex_lists = vectorized_bbox_batch_query(batch_rows, self._vertices)

            batch_time = time.time() - batch_start
            chunks_per_sec = len(batch_rows) / batch_time if batch_time > 0 else 0
            logger.debug(
                f"Batch {batch_idx + 1} completed in {batch_time:.3f} seconds ({chunks_per_sec:.0f} chunks/sec)"
            )

            # Process results for this batch
            total_vertices_found = 0
            for (idx, row), vert_mask in zip(batch_rows, vertex_lists):
                vertex_indices = np.flatnonzero(vert_mask)
                total_vertices_found += len(vertex_indices)
                id_mapping.append(
                    {
                        "l2id_idx": int(idx),
                        "vertex_mask": vertex_indices,
                    }
                )

            logger.debug(
                f"Batch {batch_idx + 1} found {total_vertices_found} vertices in {len(batch_rows)} chunks ({total_vertices_found / len(batch_rows):.1f} vertices/chunk avg)"
            )

            # Free memory from this batch
            del vertex_lists
            gc.collect()

        # Log overall performance summary
        overall_time = time.time() - overall_start
        total_chunks = len(chunk_rows)
        overall_rate = total_chunks / overall_time if overall_time > 0 else 0
        logger.info(f"Solo chunk processing complete: {total_chunks} chunks processed")
        logger.debug(
            f"Solo chunk timing: {overall_time:.3f}s ({overall_rate:.0f} chunks/sec overall)"
        )

        log_memory_usage("after all bbox_mask computation", level=logging.DEBUG)
        log_memory_usage("end of process_chunk_dataframe_solo")
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
        batch_size: int = 20,
    ) -> list:
        if self._root_id is None:
            raise ValueError(
                "Root ID must be set before processing multi-component chunks."
            )

        log_memory_usage("start of processing multi-component chunks")
        chunk_groups = [
            chunk_rows for _, chunk_rows in self._chunk_df_multi.groupby("chunk_number")
        ]
        logger.info(f"Processing {len(chunk_groups)} multi-component chunks")

        # Process chunk groups in batches to reduce joblib overhead
        total_batches = (len(chunk_groups) + batch_size - 1) // batch_size
        logger.debug(
            f"Using batched processing: {total_batches} batches of {batch_size} chunk groups each"
        )

        log_memory_usage("before parallel multicomponent processing")

        # Parallelize over chunk batches with conservative job count to avoid disk space issues
        multi_n_jobs = min(4, max(1, n_jobs)) if n_jobs != -1 else 4
        logger.debug(f"Processing with {multi_n_jobs} processes")

        all_results = []
        with tqdm(
            total=total_batches, desc="Processing multi-component chunks", unit="batch"
        ) as super_pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunk_groups))
                batch_chunk_groups = chunk_groups[start_idx:end_idx]

                logger.debug(
                    f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_chunk_groups)} chunk groups)"
                )

                # Time each batch
                import time

                batch_start = time.time()

                # Process this batch of chunk groups in parallel
                batch_results = Parallel(
                    n_jobs=multi_n_jobs,
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
                    for chunk_rows in batch_chunk_groups
                )

                batch_time = time.time() - batch_start
                chunks_per_sec = (
                    len(batch_chunk_groups) / batch_time if batch_time > 0 else 0
                )
                logger.debug(
                    f"Batch {batch_idx + 1} completed in {batch_time:.2f}s ({chunks_per_sec:.1f} chunk groups/sec)"
                )

                all_results.extend(batch_results)

                # Clean up batch memory
                del batch_results
                gc.collect()

                # Update super progress bar
                super_pbar.update(1)

        id_mapping_results = all_results

        log_memory_usage("after parallel multicomponent processing")

        # Flatten to a single list[dict]
        id_mapping: list = []
        for result in id_mapping_results:
            if isinstance(result, list) and len(result) > 0:
                id_mapping.extend(result)

        # Free the parallel processing results
        del id_mapping_results
        gc.collect()

        log_memory_usage("end of process_chunk_dataframe_multi")
        return id_mapping

    def propagate_labels(
        self,
        hop_limit: int = 50,
        batch_size: int = 500000,
    ):
        """Propagate labels using batched processing to avoid huge adjacency matrices"""
        log_memory_usage("start of batched label propagation")

        labeled_inds = np.flatnonzero(self.mesh_label_index != -1)
        unlabeled_inds = np.flatnonzero(self.mesh_label_index == -1)

        logger.info(f"Number of labeled vertices: {len(labeled_inds)}")
        logger.info(f"Number of unlabeled vertices: {len(unlabeled_inds)}")

        if len(unlabeled_inds) == 0:
            logger.info("All vertices already labeled, skipping propagation")
            return self.mesh_label

        # For small meshes, use the original method
        if len(self.vertices) < batch_size:
            logger.debug("Using original propagation method for small mesh")
            return self._propagate_labels_original(hop_limit)

        # For large meshes, use spatial batching
        logger.debug(f"Using batched propagation with batch size {batch_size}")
        return self._propagate_labels_batched(
            hop_limit, batch_size, labeled_inds, unlabeled_inds
        )

    def _propagate_labels_original(self, hop_limit: int = 50):
        """Original label propagation method for smaller meshes"""
        log_memory_usage("before adjacency matrix creation")
        A = gyp.adjacency_matrix(self.faces)
        log_memory_usage("after adjacency matrix creation")
        logger.debug(f"Adjacency matrix shape: {A.shape}, nnz: {A.nnz}")

        labeled_inds = np.flatnonzero(self.mesh_label_index != -1)

        log_memory_usage("before dijkstra computation")
        d_to, _, p2 = sparse.csgraph.dijkstra(
            A,
            indices=labeled_inds,
            limit=hop_limit,
            min_only=True,
            return_predecessors=True,
            unweighted=True,
        )
        log_memory_usage("after dijkstra computation")

        # Free the adjacency matrix as soon as we're done with it
        del A
        gc.collect()
        log_memory_usage("after freeing adjacency matrix")

        unlabeled_inds = np.flatnonzero((self.mesh_label_index == -1) & ~np.isinf(d_to))
        logger.debug(
            f"Number of vertices to propagate labels to: {len(unlabeled_inds)}"
        )
        self._mesh_label[unlabeled_inds] = self._mesh_label[p2[unlabeled_inds]]

        # Free dijkstra results
        del d_to, p2
        gc.collect()

        log_memory_usage("after original label propagation")
        return self.mesh_label

    def _propagate_labels_batched(
        self,
        hop_limit: int,
        batch_size: int,
        labeled_inds: np.ndarray,
        unlabeled_inds: np.ndarray,
    ):
        """Batched label propagation using spatial chunking"""
        log_memory_usage("start of batched label propagation")

        # Create spatial batches based on vertex positions
        num_batches = max(1, len(self.vertices) // batch_size)
        logger.debug(f"Processing label propagation in {num_batches} spatial batches")

        # Sort vertices by z-coordinate for spatial locality
        vertex_order = np.argsort(self.vertices[:, 2])
        vertices_per_batch = len(vertex_order) // num_batches

        processed_vertices = set()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * vertices_per_batch
            if batch_idx == num_batches - 1:  # Last batch gets remainder
                end_idx = len(vertex_order)
            else:
                end_idx = (batch_idx + 1) * vertices_per_batch

            batch_vertices = vertex_order[start_idx:end_idx]
            logger.debug(
                f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_vertices)} vertices)"
            )

            # Find faces that involve vertices in this batch
            faces_in_batch = np.any(np.isin(self.faces, batch_vertices), axis=1)
            batch_faces = self.faces[faces_in_batch]

            # Get all vertices involved in these faces (including neighbors)
            extended_vertices = np.unique(batch_faces)

            # Skip if all vertices in this batch are already processed
            batch_unlabeled = extended_vertices[
                np.isin(extended_vertices, unlabeled_inds)
            ]
            batch_unlabeled = batch_unlabeled[
                ~np.isin(batch_unlabeled, list(processed_vertices))
            ]

            if len(batch_unlabeled) == 0:
                logger.debug(
                    f"Batch {batch_idx + 1} has no unlabeled vertices, skipping"
                )
                continue

            # Create vertex mapping for this batch
            vertex_map = {v: i for i, v in enumerate(extended_vertices)}
            remapped_faces = np.array(
                [[vertex_map[v] for v in face] for face in batch_faces]
            )

            # Create adjacency matrix for this batch only
            log_memory_usage(f"before batch {batch_idx + 1} adjacency matrix")
            A_batch = gyp.adjacency_matrix(remapped_faces)
            log_memory_usage(f"after batch {batch_idx + 1} adjacency matrix")

            # Find labeled vertices in this batch
            batch_labeled = extended_vertices[np.isin(extended_vertices, labeled_inds)]
            batch_labeled_indices = [vertex_map[v] for v in batch_labeled]

            if len(batch_labeled_indices) == 0:
                logger.debug(
                    f"Batch {batch_idx + 1} has no labeled seed vertices, skipping"
                )
                del A_batch
                continue

            # Run dijkstra on this batch
            d_to, _, p2 = sparse.csgraph.dijkstra(
                A_batch,
                indices=batch_labeled_indices,
                limit=hop_limit,
                min_only=True,
                return_predecessors=True,
                unweighted=True,
            )

            # Map results back to original vertex indices
            batch_unlabeled_indices = [
                vertex_map[v] for v in batch_unlabeled if v in vertex_map
            ]
            reachable_mask = ~np.isinf(d_to[batch_unlabeled_indices])

            if np.any(reachable_mask):
                reachable_unlabeled_indices = np.array(batch_unlabeled_indices)[
                    reachable_mask
                ]
                predecessor_indices = p2[reachable_unlabeled_indices]

                # Update labels
                for local_unlabeled_idx, local_pred_idx in zip(
                    reachable_unlabeled_indices, predecessor_indices
                ):
                    global_unlabeled_idx = extended_vertices[local_unlabeled_idx]
                    global_pred_idx = extended_vertices[local_pred_idx]
                    self._mesh_label[global_unlabeled_idx] = self._mesh_label[
                        global_pred_idx
                    ]
                    processed_vertices.add(global_unlabeled_idx)

            # Clean up batch memory
            del A_batch, d_to, p2
            gc.collect()
            log_memory_usage(f"after batch {batch_idx + 1} cleanup")

        logger.info(
            f"Processed {len(processed_vertices)} vertices through batched propagation"
        )
        log_memory_usage("end of batched label propagation")
        return self.mesh_label

    def compute_mesh_label(
        self,
        max_distance: float = 500,
        ratio_better: float = 0.5,
        cloudvolume_fallback: bool = False,
        hop_limit: Optional[int] = None,
        coarse: bool = False,
        n_jobs: int = -1,
        solo_batch_size: int = 5000,
        propagation_batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Process the mesh.

        Returns
        -------
        np.ndarray
            Array of l2ids for each mesh label. Unassigned values have id 0.
        """
        log_memory_usage("start of compute_mesh_label")

        if hop_limit is None:
            if coarse:
                hop_limit = 75
            else:
                hop_limit = 50

        logger.info("Processing simple chunks...")
        log_memory_usage("before processing simple chunks")
        # With vectorized batch processing, we can handle large batches efficiently
        if len(self._chunk_df_solo) > 10000:
            logger.debug(
                f"Large mesh detected ({len(self._chunk_df_solo)} solo chunks), using vectorized batch processing"
            )
            # Vectorized processing can handle very large batches efficiently
            solo_batch_size = min(
                solo_batch_size, 8000
            )  # Very large batches are fine with vectorized approach

        id_mapping_solo = self.process_chunk_dataframe_solo(batch_size=solo_batch_size)
        log_memory_usage("after processing simple chunks")

        logger.info("Processing complex chunks...")
        log_memory_usage("before processing complex chunks")
        id_mapping_multi = self.process_chunk_dataframe_multi(
            cloudvolume_fallback=cloudvolume_fallback,
            max_distance=max_distance,
            ratio_better=ratio_better,
            coarse=coarse,
            n_jobs=n_jobs,
        )
        log_memory_usage("after processing complex chunks")

        mesh_label = np.full(self.vertices.shape[0], -1, dtype=int)
        for row in id_mapping_solo:
            mesh_label[row["vertex_mask"]] = row["l2id_idx"]
        for row in id_mapping_multi:
            mesh_label[row["vertex_mask"]] = row["l2id_idx"]
        self._mesh_label = mesh_label

        # Free intermediate mapping results
        del id_mapping_solo, id_mapping_multi
        gc.collect()
        log_memory_usage("after freeing intermediate mapping results")

        labeled_count = np.sum(mesh_label != -1)
        logger.info(
            f"Labeled {labeled_count}/{len(mesh_label)} vertices before propagation"
        )

        if hop_limit > 0:
            log_memory_usage("before label propagation")
            # Use much larger batch sizes since memory usage is conservative
            if propagation_batch_size is None:
                # For large meshes, use bigger batches since we have plenty of memory
                batch_size = min(1000000, max(200000, len(self.vertices) // 5))
            else:
                batch_size = propagation_batch_size
            logger.debug(f"Using propagation batch size: {batch_size}")
            self.propagate_labels(hop_limit=hop_limit, batch_size=batch_size)
            log_memory_usage("after label propagation")

        log_memory_usage("end of compute_mesh_label")
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
