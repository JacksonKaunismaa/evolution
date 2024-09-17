import time

class FPSTracker:
    """Simple class to track frames per second"""
    def __init__(self, interval=1):
        self.interval = interval
        self.start_time = time.time()
        self.frame_count = 0
        self._fps = 0

    @property
    def fps(self):
        return self._fps

    def tick(self):
        """Call this method on every frame. If the interval has passed,
        FPS will be updated."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.interval:
            fps = self.frame_count / elapsed_time
            self.start_time = time.time()
            self.frame_count = 0
            self._fps = fps