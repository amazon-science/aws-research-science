"""Shared utilities for eval scripts."""


def capture_error(e):
    """Extract all available fields from an API exception."""
    resp = getattr(e, 'response', None)
    return {
        "error_type":    type(e).__name__,
        "error_message": str(e),
        "request_id":    getattr(e, 'request_id', None),
        "status_code":   getattr(e, 'status_code', None),
        "error_code":    getattr(e, 'code', None),
        "error_param":   getattr(e, 'param', None),
        "api_error_type": getattr(e, 'type', None),
        "response_headers": dict(resp.headers) if resp is not None else None,
    }
