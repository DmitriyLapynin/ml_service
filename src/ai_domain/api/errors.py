from __future__ import annotations


class APIError(Exception):
    def __init__(self, message: str, *, status_code: int = 500, code: str = "api_error"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
