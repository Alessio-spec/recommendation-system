class InputError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, message):
        self.message = message