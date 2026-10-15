#!/usr/bin/env python3
"""Prefill missing SWE-bench image layers with resumable, SHA-256-checked GET."""
import argparse
import os
import re
import subprocess
from pathlib import Path

import requests


parser = argparse.ArgumentParser()
parser.add_argument('--images-file', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--blob-helper', type=Path, required=True)
args = parser.parse_args()
args.cache.mkdir(parents=True, exist_ok=True)
args.cache.chmod(0o700)
cache = args.cache.resolve(strict=True)
images=[x.strip() for x in args.images_file.read_text().splitlines() if x.strip() and not x.startswith('#')]
if not images:
    raise ValueError('Empty SWE-bench image list')
session=requests.Session();session.trust_env=False
for idx,image in enumerate(images,1):
    if not re.fullmatch(r'swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:latest',image):
        raise ValueError('invalid image reference')
    if subprocess.run(['docker','image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0:
        continue
    mirrored='docker.1ms.run/'+image
    repo=image.rsplit(':',1)[0]
    base='https://docker.1ms.run/v2/'+repo
    auth=session.get('https://docker.1ms.run/openapi/v1/auth/token',params={'service':'docker.1ms.run','scope':'repository:'+repo+':pull'},timeout=(5,15))
    auth.raise_for_status();token=auth.json().get('token') or auth.json().get('access_token')
    if not token:raise RuntimeError('registry token missing for '+image)
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.oci.image.index.v1+json'}
    index=session.get(base+'/manifests/latest',headers=headers,timeout=(5,15));index.raise_for_status()
    match=[x for x in index.json()['manifests'] if x.get('platform',{}).get('os')=='linux' and x.get('platform',{}).get('architecture')=='amd64']
    if len(match)!=1:raise RuntimeError('expected one linux/amd64 manifest for '+image)
    manifest=session.get(base+'/manifests/'+match[0]['digest'],headers=headers,timeout=(5,15));manifest.raise_for_status()
    layers=manifest.json()['layers'];missing=[]
    for layer in layers:
        digest=layer['digest'];expected=layer['size'];target=cache/digest
        if target.is_symlink():raise RuntimeError('symlink cache entry refused')
        if target.exists() and target.stat().st_size!=expected and target.stat().st_size!=0:
            raise RuntimeError('sized cache entry disagrees with manifest '+digest[:21])
        if not target.exists() or target.stat().st_size==0:
            missing.append((digest,expected))
    print('[{}/{}] {} missing layers {} ({} MB)'.format(idx,len(images),image,len(missing),round(sum(v for _,v in missing)/1e6)),flush=True)
    for digest,expected in missing:
        prefix=digest[7:23]
        proc=subprocess.run(['python3','-u',str(args.blob_helper),'--image',mirrored,'--digest-prefix',prefix,'--cache',str(cache),'--max-seconds','900'],timeout=940)
        if proc.returncode!=0:raise RuntimeError('verified blob download failed '+prefix)
        target=cache/digest
        if not target.is_file() or target.stat().st_size!=expected:raise RuntimeError('verified blob missing '+prefix)
print('All missing images have complete compressed blobs in user-owned cache',flush=True)
