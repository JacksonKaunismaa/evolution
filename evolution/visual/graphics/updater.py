from evolution.core.gworld import GWorld



class Updater:
    def __init__(self, gworld: GWorld, name: str):
        self.gworld = gworld
        self.name = name
        self.gworld.subscribe_updates(name)

    def update(self, *args, **kwargs):
        if self.gworld.is_updated(self.name):
            self._update(*args, **kwargs)
            self.gworld.acknowledge_update(self.name)

    def _update(self, *args, **kwargs):
        raise NotImplementedError
        