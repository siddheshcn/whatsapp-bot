
from collections import deque
from datetime import datetime
import threading

class ProgressTracker:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.progress_logs = deque(maxlen=100)  # Keep last 100 logs

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_progress(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.progress_logs.appendleft(f"[{timestamp}] {message}")

    def get_logs(self):
        return list(self.progress_logs)

def log_progress(message):
    ProgressTracker.get_instance().add_progress(message)
