"""Utility functions for the web application."""

from fastapi import Request


def is_htmx(request: Request) -> bool:
    """Check if the request was made by HTMX."""
    return request.headers.get("HX-Request") == "true"
