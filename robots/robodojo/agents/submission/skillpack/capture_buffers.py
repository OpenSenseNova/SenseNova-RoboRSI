"""Bounded, request-lifetime provider capture; normal payloads never spool to disk."""

class ResponseBuffer:
    def __init__(self, limit=16*1024*1024):
        self.limit=limit
        self.body=bytearray()
        self.overflowed=False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, value):
        if len(self.body)+len(value)>self.limit:
            self.overflowed=True
            raise ValueError('Provider response exceeded the capture memory bound')
        self.body.extend(value)
        return len(value)

    def flush(self):
        pass

    def bytes(self):
        return bytes(self.body)
