from typing import List


class Subscriber:
    def __init__(self, freq=1):
        self.freq = freq
        self.reset()
        
    def reset(self):
        self.updates_pending = 0
        self.args_pending = tuple()
        self.kwargs_pending = dict()

    def update(self):
        if self.updates_pending >= self.freq:
            self._update(*self.args_pending, **self.kwargs_pending)
            self.reset()
            
    def _update(self, *args, **kwargs):
        raise NotImplementedError
    
    def notify(self, amt, *args, **kwargs):
        self.updates_pending += amt
        self.args_pending = args
        self.kwargs_pending = kwargs
    
    
class Publisher:
    MAX_AMT = 'max'
    def __init__(self):
        self.subscribers: List[Subscriber] = []
        
    def subscribe(self, obj: Subscriber):
        if obj in self.subscribers:
            raise ValueError(f"Object {obj} already exists in subscribers dictionary.")
        self.subscribers.append(obj)
        
    def publish(self, *args, amt=1, **kwargs):
        if amt == self.MAX_AMT:
            amt = max(self.subscribers, key=lambda x: x.freq).fre
        
        for sub in self.subscribers:
            sub.notify(amt, *args, **kwargs)
            
    def init_publish(self, *args, **kwargs):
        """Force push an 'initilization' update to all subscribers."""
        amt = max(self.subscribers, key=lambda x: x.freq).freq
        for sub in self.subscribers:
            sub.notify(amt, *args, **kwargs)
    
    def unsubscribe(self, obj: Subscriber):
        self.subscribers.remove(obj)
        