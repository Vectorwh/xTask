"""Redact registered runtime credentials, including encoded command wrappers."""
import base64
import logging
import threading
import shlex

_VALUES = set()
_LOCK = threading.Lock()


def redact(value):
    text = str(value)
    with _LOCK:
        values = tuple(_VALUES)
    for secret in sorted(values, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    return text


class _SecretFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return True


def register_secret(value):
    if not value:
        return
    with _LOCK:
        _VALUES.add(value)
        _VALUES.add(shlex.quote(value))
    logger = logging.getLogger()
    # Root logger filters do not apply to propagated child records; handlers do.
    for handler in logger.handlers:
        if not any(isinstance(item, _SecretFilter) for item in handler.filters):
            handler.addFilter(_SecretFilter())


def register_encoded_command(command):
    with _LOCK:
        contains_secret = any(value in command for value in _VALUES)
    if contains_secret:
        encoded = base64.b64encode(command.encode())
        register_secret(encoded.decode())
        register_secret(repr(list(encoded)))


def redact_tree(value):
    """Sanitize a report copy, never mutate live authentication configuration."""
    if isinstance(value, dict):
        return {key: ("<redacted>" if str(key).lower() in {
                    "internal_api_token", "execution_termination_api_token", "xllm_internal_token", "api_key"
                } else redact_tree(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_tree(item) for item in value)
    return redact(value) if isinstance(value, str) else value
