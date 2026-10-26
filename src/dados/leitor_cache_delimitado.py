"""Read one fixed cache value per admitted key, without any clinical reference.

This is a real-data adapter, separate from synthetic scientific modules. The
caller must install the process-level file/network/connection guard and verify
the queue manifest before constructing it. Python reflection is not a sandbox.
No SQL, connection, cursor or raw value is included in the public audit report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import sqlite3


VERSION = 'restricted-cache-reader-v1'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_QUERY = 'SELECT v FROM r WHERE k = ?'


class CacheAccessViolation(RuntimeError):
    """The attempted operation is outside the admitted read-only scope."""


class CacheIntegrityError(RuntimeError):
    """The pinned, standalone cache cannot be verified."""


class CacheCardinalityError(RuntimeError):
    """More than one value belongs to a supposedly unique admitted key."""


@dataclass(frozen=True)
class CacheReadResult:
    status: str
    value: object = field(default=None, repr=False)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _keys_digest(keys):
    return hashlib.sha256(('\n'.join(sorted(keys)) + '\n').encode('ascii')).hexdigest()


def cache_uri(path):
    """Exact URI for the process guard's one-connection allowlist."""
    return Path(path).resolve().as_uri() + '?mode=ro&immutable=1'


class RestrictCacheReader:
    """Context-managed fixed query, immutable/ro, at most one read per key.

    Absence is a distinct result, not evidence of a provider failure. A present
    NULL or malformed serialized value is handed to the parser as present.
    Hashes are checked on entry and exit; an altered file fails the whole pass.
    The SQLite authorizer is an additional barrier, not the key admission rule.
    """
    def __init__(self, path, allowed_keys, expected_sha256):
        keys = list(allowed_keys)
        if (not keys or any(not isinstance(k, str) or not _HEX.fullmatch(k) for k in keys)
                or len(set(keys)) != len(keys)):
            raise CacheAccessViolation('Expected a nonempty unique admitted key list')
        if not isinstance(expected_sha256, str) or not _HEX.fullmatch(expected_sha256):
            raise CacheIntegrityError('Invalid pinned cache digest')
        self.__path = Path(path).resolve()
        self.__keys = frozenset(keys)
        self.__expected = expected_sha256
        self.__connection = None
        self.__started = False
        self.__closed = False
        self.__query_active = False
        self.__used = set()
        self.__before = None
        self.__after = None
        self.__present = 0
        self.__absent = 0
        self.__rejected_keys = 0
        self.__duplicate_attempts = 0
        self.__cardinality_failures = 0
        self.__sql_failures = 0
        self.__sql_denials = 0
        self.__integrity_failure = False

    def __sidecars_absent(self):
        return not any(Path(str(self.__path) + suffix).exists()
                       for suffix in ('-wal', '-shm', '-journal'))

    def __authorize(self, action, first, second, database, source):
        permitted = self.__query_active and source is None and (
            (action == sqlite3.SQLITE_SELECT and database is None)
            or (action == sqlite3.SQLITE_READ and first == 'r'
                and second in {'k', 'v'} and database == 'main'))
        if not permitted:
            self.__sql_denials += 1
        return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY

    def __enter__(self):
        if self.__started or self.__closed:
            raise CacheAccessViolation('Reader cannot be reopened')
        self.__started = True
        try:
            if not self.__path.is_file() or not self.__sidecars_absent():
                raise CacheIntegrityError('Standalone cache required; sidecars are forbidden')
            self.__before = _file_digest(self.__path)
            if self.__before != self.__expected:
                raise CacheIntegrityError('Cache digest differs from the fixed input')
            self.__connection = sqlite3.connect(
                cache_uri(self.__path), uri=True,
                cached_statements=0, check_same_thread=True)
            self.__connection.set_authorizer(self.__authorize)
        except (OSError, sqlite3.Error) as exc:
            self.__integrity_failure = True
            self.__closed = True
            if self.__connection is not None:
                self.__connection.close()
                self.__connection = None
            raise CacheIntegrityError('Unable to open verified read-only cache') from None
        except CacheIntegrityError:
            self.__integrity_failure = True
            self.__closed = True
            raise
        return self

    def read_value(self, key):
        """Return the sole value/absence for an admitted key; never execute SQL input."""
        if self.__connection is None or self.__closed:
            raise CacheAccessViolation('Reader is not open')
        if not isinstance(key, str) or key not in self.__keys:
            self.__rejected_keys += 1
            raise CacheAccessViolation('Key is outside the fixed queue')
        if key in self.__used:
            self.__duplicate_attempts += 1
            raise CacheAccessViolation('Key has already been attempted')
        self.__used.add(key)
        cursor = None
        try:
            self.__query_active = True
            cursor = self.__connection.execute(_QUERY, (key,))
            rows = cursor.fetchmany(2)
        except sqlite3.Error:
            self.__sql_failures += 1
            raise CacheAccessViolation('Fixed cache query failed') from None
        finally:
            self.__query_active = False
            if cursor is not None:
                cursor.close()
        if len(rows) > 1:
            self.__cardinality_failures += 1
            raise CacheCardinalityError('Multiple results for an admitted key')
        if not rows:
            self.__absent += 1
            return CacheReadResult('absent')
        self.__present += 1
        return CacheReadResult('present', rows[0][0])

    def __exit__(self, exc_type, exc, traceback):
        if self.__connection is not None:
            self.__connection.close()
            self.__connection = None
        self.__closed = True
        try:
            self.__after = _file_digest(self.__path)
            if not self.__sidecars_absent() or self.__after != self.__expected:
                raise CacheIntegrityError('Cache or sidecar state changed during the pass')
        except (OSError, CacheIntegrityError):
            self.__integrity_failure = True
            raise CacheIntegrityError('Final cache integrity verification failed') from None
        return False

    def report(self):
        """Safe counters and digests; no keys, paths, SQL parameters or values."""
        return dict(contract_version=VERSION, read_only=True, immutable=True,
                    query_template=_QUERY, allowed_keys=len(self.__keys),
                    allowed_keys_sha256=_keys_digest(self.__keys),
                    attempted_unique_keys=len(self.__used),
                    attempted_keys_sha256=_keys_digest(self.__used),
                    present_values=self.__present, absent_keys=self.__absent,
                    rejected_keys=self.__rejected_keys,
                    duplicate_read_attempts=self.__duplicate_attempts,
                    cardinality_failures=self.__cardinality_failures,
                    sql_failures=self.__sql_failures, sql_denials=self.__sql_denials,
                    expected_cache_sha256=self.__expected,
                    cache_sha256_before=self.__before, cache_sha256_after=self.__after,
                    cache_unchanged=(self.__before == self.__expected == self.__after),
                    integrity_failure=self.__integrity_failure, closed=self.__closed,
                    raw_values_in_report=False)
