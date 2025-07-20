class BaseError(Exception):
    """Base class for all custom exceptions in the application."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message