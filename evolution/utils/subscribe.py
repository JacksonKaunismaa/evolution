from typing import Any, List


class Subscriber:
    def __init__(self, freq=1):
        self.freq = freq
        self.reset()

    def reset(self):
        self.updates_pending = 0
        self.args_pending = tuple()
        self.kwargs_pending = dict()

    def update(self) -> Any:
        """Update the subscriber if the number of updates pending is greater than or equal to the frequency.

        Returns:
            Whatever _update returns (if any)
        """
        if self.updates_pending >= self.freq:
            retval = self._update(*self.args_pending, **self.kwargs_pending)
            self.reset()
            return retval

    def _update(self, *args, **kwargs):
        raise NotImplementedError

    def notify(self, amt, *args, **kwargs):
        self.updates_pending += amt
        self.args_pending = args
        self.kwargs_pending = kwargs


class Publisher:
    def __init__(self):
        self.subscribers: List[Subscriber] = []

    def subscribe(self, obj: Subscriber):
        if obj in self.subscribers:
            raise ValueError(f"Object {obj} already exists in subscribers dictionary.")
        self.subscribers.append(obj)

    def publish(self, *args, amt=1, **kwargs):
        for sub in self.subscribers:
            sub.notify(amt, *args, **kwargs)

    def init_publish(self, *args, **kwargs):
        """Force push an 'initilization' update to all subscribers."""
        if not self.subscribers:
            return
        amt = max(self.subscribers, key=lambda x: x.freq).freq
        self.publish(*args, amt=amt, **kwargs)

    def update_all(self):
        for sub in self.subscribers:
            sub.update()

    def unsubscribe(self, obj: Subscriber):
        self.subscribers.remove(obj)
