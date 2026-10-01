"""Minimal loader for future resolution gates; it never selects or samples inputs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reports/superpoint_resolution_gate/r0_protocol_manifest.json'


def load_r0_protocol(path: Path = MANIFEST) -> dict:
    """Load the frozen R0 protocol. Caller must use its ordered_images/matrices verbatim."""
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('protocol_version') != 'SP-K210-R0-frozen-v1':
        raise ValueError('Unsupported or missing R0 protocol manifest version')
    if manifest.get('image_count') != 30 or len(manifest.get('ordered_images', [])) != 30:
        raise ValueError('R0 manifest image list is incomplete')
    if manifest.get('topk_values') != [256, 512, 1024]:
        raise ValueError('R0 manifest Top-K settings differ from frozen protocol')
    return manifest


def homography_for(manifest: dict, image_index: int, resolution: str) -> list:
    """Return the stored matrix verbatim; do not regenerate it from a seed or formula."""
    entry = manifest['homographies'][image_index]
    if entry['image_index'] != image_index:
        raise ValueError('Manifest image/homography ordering is inconsistent')
    try:
        return entry['matrices_source_to_warped_xy_pixels'][resolution]
    except KeyError as error:
        raise ValueError('No frozen matrix for resolution {}'.format(resolution)) from error
