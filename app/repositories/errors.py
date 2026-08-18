class NotFoundError(Exception):
    """Raised when a repository is asked to load a row that doesn't exist."""

    def __init__(self, entity: str, id: str | int):
        self.entity = entity
        self.id = id
        super().__init__(f"{entity} not found: {id!r}")
