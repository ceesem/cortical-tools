# from cortexclient.mesh_vertex import *
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("mesh_processing.log"),
#         logging.StreamHandler(),  # Also print to console
#     ],
# )

from cortical_tools.datasets.v1dd import client

rid = 864691132544823185

# from cortical_tools.datasets.microns_prod import client

# rid = 864691135101289504
import time

if __name__ == "__main__":
    t0 = time.time()
    l2mapping = client.mesh.compute_vertex_to_l2_mapping(root_id=rid)
    logging.warning(f"Time taken: {time.time() - t0:.2f} seconds")
    print(len(l2mapping), "vertices")
