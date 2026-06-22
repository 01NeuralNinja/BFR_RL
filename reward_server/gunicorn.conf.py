import os

# Specify here all the GPU IDs you want to use.
AVAILABLE_DEVICES = [0]

# Track which devices are already taken by workers.
USED_DEVICES = set()

# Port the service listens on.
port = 18085

def pre_fork(server, worker):
    """
    Runs in the master process before a worker is forked.
    Assigns an available GPU ID to the new worker.
    """
    global USED_DEVICES
    # Pick the first device in the list that is not yet in use.
    worker.device_id = next(i for i in AVAILABLE_DEVICES if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)
    server.log.info(f"Worker {worker.pid} assigned to GPU {worker.device_id}")

def post_fork(server, worker):
    """
    Runs in the worker process after it is forked.
    Sets CUDA_VISIBLE_DEVICES for this specific worker.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)

def child_exit(server, worker):
    """
    Runs in the master process when a worker exits.
    Frees the GPU held by that worker so new processes can reuse it.
    """
    global USED_DEVICES
    if hasattr(worker, 'device_id') and worker.device_id in USED_DEVICES:
        USED_DEVICES.remove(worker.device_id)
        server.log.info(f"Released GPU {worker.device_id} from worker {worker.pid}")

# --- Gunicorn configuration ---
bind = f"127.0.0.1:{port}"

# Number of workers should equal the number of GPUs specified.
workers = len(AVAILABLE_DEVICES)

worker_class = "sync"
timeout = 300
# Verbose log level so allocation info is visible in the terminal.
loglevel = "info"

limit_request_body = 104857600
