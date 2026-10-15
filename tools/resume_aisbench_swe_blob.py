#!/usr/bin/env python3
"""Resume exactly one OCI compressed image layer into a user-owned crane cache."""
import argparse
import hashlib
import os
import re
import time
from pathlib import Path

import requests
import urllib3


def fail(message):
    raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--digest-prefix', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--proxy', default=os.environ.get('HTTPS_PROXY', ''))
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    if not re.fullmatch(r'docker\.1ms\.run/swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:latest', args.image):
        fail('unexpected image name')
    if not re.fullmatch(r'[0-9a-f]{12,64}', args.digest_prefix):
        fail('invalid digest prefix')
    repo = args.image.split('/', 1)[1].rsplit(':', 1)[0]
    base = 'https://docker.1ms.run/v2/' + repo
    meta = requests.Session()
    meta.trust_env = False
    response = meta.get('https://docker.1ms.run/openapi/v1/auth/token',
                        params={'service': 'docker.1ms.run', 'scope': 'repository:' + repo + ':pull'}, timeout=(5, 15))
    response.raise_for_status()
    bearer = response.json().get('token') or response.json().get('access_token')
    if not bearer:
        fail('mirror token missing')
    headers = {'Authorization': 'Bearer ' + bearer,
               'Accept': 'application/vnd.oci.image.manifest.v1+json,application/vnd.oci.image.index.v1+json'}
    index = meta.get(base + '/manifests/latest', headers=headers, timeout=(5, 15))
    index.raise_for_status()
    desc = next((m for m in index.json()['manifests'] if m.get('platform', {}).get('os') == 'linux'
                 and m.get('platform', {}).get('architecture') == 'amd64'), None)
    if desc is None:
        fail('linux/amd64 manifest missing')
    manifest = meta.get(base + '/manifests/' + desc['digest'], headers=headers, timeout=(5, 15))
    manifest.raise_for_status()
    matches = [l for l in manifest.json()['layers'] if l['digest'].startswith('sha256:' + args.digest_prefix)]
    if len(matches) != 1:
        fail('expected exactly one matching image layer')
    digest = matches[0]['digest']
    expected_size = int(matches[0]['size'])
    args.cache.mkdir(parents=True, exist_ok=True)
    args.cache.chmod(0o700)
    cache = args.cache.resolve(strict=True)
    target = cache / digest
    partial = cache / (digest + '.partial')
    if target.is_symlink() or partial.is_symlink():
        fail('refusing symlink in cache')
    if target.exists() and target.stat().st_size == expected_size:
        print('target already has expected size; checking sha256', flush=True)
        path = target
    else:
        if target.exists():
            if target.stat().st_size != 0:
                fail('nonempty cache entry with unexpected size')
            target.unlink()
        path = partial
        if path.exists() and path.stat().st_size > expected_size:
            fail('partial file is larger than manifest')
        deadline = time.monotonic() + args.max_seconds
        for attempt in range(1, 13):
            offset = path.stat().st_size if path.exists() else 0
            if offset >= expected_size:
                break
            if time.monotonic() > deadline:
                fail('download deadline reached')
            blob = meta.get(base + '/blobs/' + digest, headers=headers, allow_redirects=False,
                            timeout=(5, 15))
            if blob.status_code != 307 or not blob.headers.get('Location'):
                fail('mirror did not return a blob redirect (HTTP {})'.format(blob.status_code))
            # Only proxy the blob CDN, not the registry API; the proxy permits
            # resumable range requests even when a long-lived stream is reset.
            route = 'proxy' if attempt % 2 else 'direct'
            session = requests.Session()
            session.trust_env = False
            if route == 'proxy' and args.proxy:
                session.proxies = {'http': args.proxy, 'https': args.proxy}
            try:
                with session.get(blob.headers['Location'], headers={'Range': 'bytes={}-'.format(offset),
                                  'Accept-Encoding': 'identity'}, timeout=(6, 20), stream=True) as data:
                    if data.status_code != 206:
                        fail('CDN does not support a resumable range (HTTP {})'.format(data.status_code))
                    content_range = data.headers.get('Content-Range', '')
                    if not content_range.startswith('bytes {}-'.format(offset)) or not content_range.endswith('/{}'.format(expected_size)):
                        fail('CDN responded with inconsistent content range')
                    with path.open('ab') as output:
                        os.chmod(path, 0o600)
                        for chunk in data.raw.stream(1024 * 1024, decode_content=False):
                            if chunk:
                                output.write(chunk)
                                offset += len(chunk)
                                if offset > expected_size:
                                    fail('CDN sent more bytes than manifest specifies')
                            if time.monotonic() > deadline:
                                fail('download deadline reached')
                print('attempt {} {}: {:.1f}/{:.1f} MB'.format(attempt, route, offset/1e6, expected_size/1e6), flush=True)
            except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, OSError) as exc:
                print('attempt {} {} interrupted: {} at {:.1f}/{:.1f} MB'.format(
                    attempt, route, type(exc).__name__, offset/1e6, expected_size/1e6), flush=True)
            finally:
                session.close()
        if path.stat().st_size != expected_size:
            fail('download incomplete: {} of {} bytes'.format(path.stat().st_size, expected_size))
    actual = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            actual.update(chunk)
    if actual.hexdigest() != digest.split(':', 1)[1]:
        fail('SHA-256 mismatch; refusing to cache blob')
    if path != target:
        os.replace(path, target)
    print('VERIFIED {} bytes SHA256 {}...'.format(expected_size, digest[7:23]), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
        # Signed blob URLs can include credentials; do not echo exception URLs.
        print('SWE layer download failed: {}'.format(type(exc).__name__), flush=True)
        raise SystemExit(1)
