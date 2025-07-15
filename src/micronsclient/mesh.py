from numbers import Integral

import numpy as np
from trimesh import Trimesh


def _convert_to_trimesh(mesh):
    """
    Convert a mesh to a trimesh object.
    """
    if isinstance(mesh, Trimesh):
        return mesh
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    else:
        raise ValueError(
            "Invalid mesh format. Must be a Trimesh object or have vertices and faces attributes."
        )


class MeshClient:
    def __init__(
        self,
        caveclient=None,
    ):
        self._cv = None
        self._cc = caveclient

    @property
    def cv(self):
        if self._cv is None:
            self._build_cv()
        return self._cv

    def _build_cv(self):
        self._cv = self._cc.info.segmentation_cloudvolume()

    def _get_meshes(self, root_ids, progress):
        curr_prog = self.cv.progress is True
        self._cv.progress = progress
        meshes = self.cv.mesh.get(root_ids, allow_missing=False)
        self._cv.progress = curr_prog
        return meshes

    def get_mesh(
        self,
        root_id: int,
        *,
        progress: bool = False,
        as_trimesh: bool = False,
    ):
        """Get single mesh from root id

        Parameters
        ----------
        root_id : int
            Root ID for a neuron
        progress : bool, optional
            If True, use progress bar, by default True
        as_trimesh : bool, optional
            If True, converts to a trimesh Trimesh object, by default False

        Returns
        -------
        Mesh
            Mesh
        """
        if not isinstance(root_id, Integral):
            raise ValueError("This function takes only one root id")

        mesh = self._get_meshes(root_id, progress)[root_id]
        if as_trimesh:
            return _convert_to_trimesh(mesh)
        else:
            return mesh

    def get_meshes(
        self,
        root_ids: list,
        *,
        progress: bool = True,
        as_trimesh: bool = False,
    ):
        """Get multiple meshes from root ids.

        Parameters
        ----------
        root_ids : list
            List of root ids
        progress : bool, optional
            If True, use progress bar, by default True
        as_trimesh : bool, optional
            If True, converts each mesh to a trimesh Trimesh object, by default False
        """
        meshes = self._get_meshes(root_ids, progress)
        if as_trimesh:
            return {
                root_id: _convert_to_trimesh(meshes[root_id]) for root_id in root_ids
            }
        else:
            return meshes
