import uuid

UNDEFINED = object()

class ID(object):

    def __init__(self, id: uuid.UUID=None):

        self.x = id if id is not None else uuid.uuid4()
