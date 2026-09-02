# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""A small connection pool shared by every dashboard API request.

The dashboard is read-mostly and request-driven (unlike the storage
writer's steady background stream), so a bounded pool that hands a
connection to each request and returns it afterward -- rather than a single
shared connection -- is what keeps concurrent requests from serializing on
one socket.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg2.pool import SimpleConnectionPool

from src.config.constants import TimeoutConstants

_MIN_POOL_CONNECTIONS = 1
_MAX_POOL_CONNECTIONS = 10

_pool: SimpleConnectionPool | None = None


def init_pool(dsn: str) -> None:
    global _pool
    _pool = SimpleConnectionPool(
        _MIN_POOL_CONNECTIONS,
        _MAX_POOL_CONNECTIONS,
        dsn,
        connect_timeout=TimeoutConstants.DB_CONNECT_TIMEOUT_SECONDS,
    )


@contextmanager
def get_connection() -> Iterator[object]:
    if _pool is None:
        raise RuntimeError("Connection pool not initialized; call init_pool() at startup.")
    connection = _pool.getconn()
    try:
        yield connection
    finally:
        _pool.putconn(connection)
