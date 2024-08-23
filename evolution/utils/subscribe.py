from typing import List


class Subscriber:
    def __init__(self, freq=1):
        self.freq = freq
        self.updates_pending = 0

    def update(self, *args, **kwargs):
        if self.updates_pending >= self.freq:
            self._update(*args, **kwargs)
            self.updates_pending = 0

    def _update(self, *args, **kwargs):
        raise NotImplementedError
    
    def notify(self, amt=1):
        self.updates_pending += amt
    
    
class Publisher:
    def __init__(self):
        self.subscribers: List[Subscriber] = []
        
    def subscribe(self, obj: Subscriber):
        if obj in self.subscribers:
            raise ValueError(f"Object {obj} already exists in subscribers dictionary.")
        self.subscribers.append(obj)
        obj.notify(obj.freq)
        
    def publish(self):
        for sub in self.subscribers:
            sub.notify()
    
    def unsubscribe(self, obj: Subscriber):
        self.subscribers.remove(obj)
        