import threading
import time

import GPUtil


class ContinuousGPUMonitor:
    """Continuously monitors GPU usage over time using a separate thread that
    periodically queries GPU information and store it at regular intervals."""

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.stop_thread = threading.Event()
        self.log_thread = threading.Thread(target=self.log_gpu_usage)
        self.gpu_data = {"gpu_memory": []}

    def log_gpu_usage(self) -> None:
        while not self.stop_thread.is_set():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_memory_usage = gpu.memoryUsed
                self.gpu_data["gpu_memory"].append(gpu_memory_usage)

            time.sleep(self.interval)

    def __enter__(self):
        self.log_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_thread.set()
        self.log_thread.join()
