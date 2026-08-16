"""Shared test doubles."""


class FakeBlob:
    """In-memory stand-in for a google-cloud-storage blob."""

    def __init__(self, store, name):
        self.store, self.name = store, name
        self.cache_control = None

    def exists(self):
        return self.name in self.store

    def download_as_bytes(self):
        if self.name not in self.store:
            raise FileNotFoundError(f"404 Not Found: {self.name}")
        return self.store[self.name]

    def upload_from_string(self, data, content_type=None):
        self.store[self.name] = (data.encode() if isinstance(data, str)
                                 else bytes(data))

    def make_public(self):
        pass


class FakeBucket:
    """Dict-backed bucket: enough surface for adapters + training_sync."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def blob(self, name):
        return FakeBlob(self.store, name)
