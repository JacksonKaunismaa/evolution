import time

class FPSTracker:
    def __init__(self, interval=1):
        self.interval = interval
        self.start_time = time.time()
        self.frame_count = 0
        self._fps = 0
        
    @property
    def fps(self):
        return self._fps
        
    def tick(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.interval:
            fps = self.frame_count / elapsed_time
            self.start_time = time.time()
            self.frame_count = 0
            self._fps = fps