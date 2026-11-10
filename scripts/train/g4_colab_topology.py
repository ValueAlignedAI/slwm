"""Run through the official notebook API, then source the hardware settings in the same VM's SSH shell."""
import os
from pathlib import Path
import shlex

KEYS=("TPU_ACCELERATOR_TYPE","TPU_HOST_BOUNDS","TPU_CHIPS_PER_HOST_BOUNDS","TPU_PROCESS_BOUNDS",
      "TPU_CHIPS_PER_PROCESS_BOUNDS","TPU_VISIBLE_CHIPS","TPU_SKIP_MDS_QUERY","TPU_WORKER_ID",
      "CLOUD_TPU_TASK_ID","TPU_MESH_CONTROLLER_ADDRESS","TPU_MESH_CONTROLLER_PORT","PJRT_DEVICE",
      "TPU_PROCESS_ADDRESSES","TPU_PROCESS_PORT","TPU_PROCESS_TASK_ID","TPU_WORKER_HOSTNAMES","TPU_NAME","TPU_TOPOLOGY")
settings={k:os.environ[k] for k in KEYS if k in os.environ}
if "TPU_ACCELERATOR_TYPE" not in settings:
    raise RuntimeError("This notebook kernel did not advertise TPU hardware")
dest=Path("/content/g4-topology.env")
dest.write_text("".join(f"export {k}={shlex.quote(v)}\n" for k,v in settings.items()))
dest.chmod(0o600)
print("Configured TPU hardware",settings["TPU_ACCELERATOR_TYPE"])
