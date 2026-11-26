"""Verify node-local VLA handshakes against the pinned model and startup receipt."""
from __future__ import annotations

import inspect
import json
from pathlib import Path


def pi05_identity(policy_dir: Path, *, config_name: str | None = None) -> dict:
    """Describe the authenticated model inputs; loading remains the server's job."""
    if __package__:
        from .pi05_assets import load_pi05_metadata
        from .pi05_source import PI05_SOURCE_MANIFEST_SHA256
    else:
        from pi05_assets import load_pi05_metadata
        from pi05_source import PI05_SOURCE_MANIFEST_SHA256
    manifest = load_pi05_metadata(policy_dir)
    if config_name is not None and config_name != manifest['config_name']:
        raise ValueError('pi05 config must select the original ARX-X5 step59999 policy')
    files = {row['path']: row['sha256'] for row in manifest['files']}
    return {'backend': 'pi05', 'config_name': manifest['config_name'],
            'checkpoint_metadata_sha256': files['_CHECKPOINT_METADATA'],
            'norm_stats_sha256': files['assets/arx_x5_sim/norm_stats.json'],
            'source_manifest_sha256': PI05_SOURCE_MANIFEST_SHA256}


def validate_shared_metadata(metadata: dict, *, backend: str, port: int,
                             receipt_path: Path | None, policy_dir: Path) -> None:
    """Refuse a different model or a stale receipt before any episode runs."""
    if backend == 'simate_beta':
        if metadata.get('policy') != 'Simate_beta':
            raise ValueError('Shared VLA identity is not Simate_beta')
        return
    if receipt_path is None:
        raise ValueError('Shared pi05 requires startup receipt ROBORSI_SHARED_VLA_RECEIPT from Pi05Server startup; restart legacy services with the current wrapper')
    path = Path(receipt_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Shared pi05 receipt must be an existing absolute regular file')
    receipt = json.loads(path.read_text())
    actual = metadata.get('roborsi_vla_identity')
    expected = pi05_identity(policy_dir)
    if (not isinstance(actual, dict) or any(actual.get(k) != v for k,v in expected.items())
            or not isinstance(actual.get('service_id'), str) or not actual['service_id']):
        raise ValueError('Shared pi05 identity differs from the original ARX-X5 step59999 model')
    if (receipt.get('schema_version') != 1 or receipt.get('port') != port
            or receipt.get('identity') != actual
            or receipt.get('checkpoint_files_verified') is not True):
        raise ValueError('Shared pi05 receipt does not match the running service identity')


def check_shared_vla(port: int, *, backend: str, receipt_path: Path | None,
                     policy_dir: Path) -> None:
    """Read a real loopback WebSocket handshake without requesting inference."""
    from websockets.sync.client import connect
    from openpi_client import msgpack_numpy
    options = {'proxy': None} if 'proxy' in inspect.signature(connect).parameters else {}
    with connect(f'ws://127.0.0.1:{port}', compression=None, max_size=65536,
                 open_timeout=5, close_timeout=5, **options) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=5))
        if not isinstance(metadata, dict):
            raise ValueError('Shared VLA identity metadata must be an object')
        validate_shared_metadata(metadata, backend=backend, port=port,
                                 receipt_path=receipt_path, policy_dir=policy_dir)
