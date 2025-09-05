from numbers import Integral
from typing import TYPE_CHECKING, Optional

import numpy.typing as npt

if TYPE_CHECKING:
    from caveclient import CAVEclientFull

import logging

import numpy as np
import urllib3
from cloudvolume import CloudVolume

from .mesh_vertex import VertexAssigner

urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class MeshClient:
    def __init__(
        self,
        caveclient: Optional["CAVEclientFull"] = None,
        cv_path: Optional[str] = None,
    ):
        self._cv = None
        self._cc = caveclient
        self._cv_path = cv_path
        self._mesh_labels = dict()

    @property
    def cv(self):
        if self._cv is None:
            self._build_cv()
        return self._cv

    def _build_cv(self):
        if self._cv_path is None:
            self._cv = self._cc.info.segmentation_cloudvolume()
        else:
            self._cv = CloudVolume(self._cv_path, progress=False, use_https=True)

    def _get_meshes(self, root_ids, progress, **kwargs):
        curr_prog = self.cv.progress is True
        self._cv.progress = progress
        if "fuse" in kwargs:
            del kwargs["fuse"]
        meshes = self.cv.mesh.get(root_ids, fuse=False, **kwargs)
        self._cv.progress = curr_prog
        return meshes

    def get_mesh(
        self,
        root_id: int,
        *,
        progress: bool = False,
        **kwargs,
    ):
        """Get single mesh from root id

        Parameters
        ----------
        root_id : int
            Root ID for a neuron
        progress : bool, optional
            If True, use progress bar, by default True
        kwargs: dict, optional
            Additional keyword arguments to pass to cloudvolume.mesh.get.

        Returns
        -------
        Mesh
            Mesh
        """
        if not isinstance(root_id, Integral):
            raise ValueError("This function takes only one root id")
        mesh = self._get_meshes(root_id, progress, **kwargs).get(root_id)
        return mesh

    def get_meshes(
        self,
        root_ids: list,
        *,
        progress: bool = True,
        **kwargs,
    ):
        """Get multiple meshes from root ids.

        Parameters
        ----------
        root_ids : list
            List of root ids
        progress : bool, optional
            If True, use progress bar, by default True

        kwargs: dict, optional
            Additional keyword arguments to pass to cloudvolume.mesh.get.

        Returns
        -------
        dict
            Dictionary of meshes keyed by root id.
        """
        meshes = self._get_meshes(root_ids, progress, **kwargs)
        return meshes

    def compute_vertex_to_l2_mapping(
        self,
        root_id: int,
        vertices: Optional[npt.NDArray] = None,
        faces: Optional[npt.NDArray] = None,
        lvl2_ids: Optional[npt.NDArray] = None,
        lvl2_pts: Optional[npt.NDArray] = None,
        max_distance: float = 500,
        ratio_better: float = 0.5,
        hop_limit: Optional[int] = None,
        cloudvolume_fallback: bool = False,
        n_jobs: int = -1,
        return_assigner: bool = False,
        mesh_kwargs: Optional[dict] = None,
    ) -> npt.NDArray:
        """Compute an approximate mapping for each mesh vertex to the associated layer 2 id.
        Note that this is close but somewhat heuristic due to the nature of how meshes are produced.
        Assignment is based first on the representative points of layer 2 ids, and then falls back to heuristic methods for floating mesh components.
        If a vertex cannot be assigned a layer 2 id, it will be assigned a default value of 0.

        Parameters
        ----------
        root_id : int
            Root ID for a neuron
        vertices : Optional[npt.NDArray]
            Vertex positions, if you have a mesh already downloaded. Will be downloaded otherwise.
        faces : Optional[npt.NDArray]
            Face indices, if you have a mesh already downloaded. Will be downloaded otherwise.
        lvl2_ids : Optional[npt.NDArray]
            Layer 2 IDs, if already loaded. Will be downloaded otherwise.
        lvl2_pts : Optional[npt.NDArray]
            Layer 2 points, if already loaded. Will be downloaded otherwise.
        max_distance : float
            Maximum distance for mesh compartment assignment based on proximity, in nanometers (or mesh units).
        ratio_better : float
            Ratio for how much better a proximity-based assignment must be than the second-best assignment to be used.
        hop_limit : Optional[int]
            Hop limit for assignment of unassigned nodes via closest graph traversal.
        cloudvolume_fallback : bool
            Use CloudVolume to try to download data. Much slower.
        n_jobs : int
            Number of jobs for parallel processing. Defaults to -1 (all available cores).
        return_assigner: bool
            If True, returns the vertex assigner object as well. Defaults to False.
        mesh_kwargs: dict
            Additional keyword arguments to pass to the mesh download function.

        Returns
        -------
        l2_mapping: npt.NDArray
            Array of layer 2 IDs for each vertex.
        vertex_assigner: mesh_vertex.VertexAssigner
            Object containing information about the vertex assignment process, including mesh vertices and faces.
        """
        if self._cc is None:
            raise ValueError("CAVE client is not set. Please provide a CAVE client.")

        va = VertexAssigner(
            root_id=root_id,
            caveclient=self._cc,
            vertices=vertices,
            faces=faces,
            lvl2_ids=lvl2_ids,
            lvl2_pts=lvl2_pts,
        )
        mesh_labels = va.compute_mesh_label(
            max_distance=max_distance,
            ratio_better=ratio_better,
            cloudvolume_fallback=cloudvolume_fallback,
            hop_limit=hop_limit,
            n_jobs=n_jobs,
        )
        self._mesh_labels[root_id] = mesh_labels
        if return_assigner:
            return mesh_labels, va
        else:
            return mesh_labels

    @property
    def mesh_l2_mappings(self) -> dict:
        """Get the dictionary of root id to mesh vertex to layer 2 id mappings."""
        return self._mesh_labels
