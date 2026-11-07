"""B01 CPU experiment: unchanged baseline training, two checkpoint criteria."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from neuroforge.models.neural.checkpoint_selection import CheckpointSelection
from neuroforge.models.neural.progress import install_progress, peak_ram_gib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/B01_cpu_smoke.json')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if config.get('device') != 'cpu':
        raise ValueError('This initial B01 runner uses CPU')
    params = config['params']
    if params['train_feeds'] <= 0 or params['val_feeds'] <= 0:
        raise ValueError('Use bounded whole-feed samples until memory and development split are approved')
    spec = importlib.util.spec_from_file_location('upstream_b01', ROOT / 'baseline/neural_baseline.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    baseline.TRAIN_CONFIG.update(params)
    install_progress(baseline)
    baseline.save_epoch = CheckpointSelection(baseline)
    started = time.perf_counter()
    baseline.main(args=['--device', 'cpu', 'train', '--data', str(args.data),
                        '--output', str(args.output)], standalone_mode=False)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    files = ['baseline/neural_baseline.py', 'src/neuroforge/metrics.py',
             'src/neuroforge/models/neural/checkpoint_selection.py', 'scripts/train_b01.py']
    manifest = {'commit': commit, 'working_tree_status': status, 'python': platform.python_version(),
                'torch': torch.__version__, 'seconds': time.perf_counter() - started,
                'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
                'data': str(args.data.resolve()), 'selection': 'First whole feeds in sorted files; smoke only',
                'peak_ram_gib': peak_ram_gib(), 'vram': 'CPU run; not applicable'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'report.md').write_text(
        '# B01 smoke\n\nSame architecture, loss, optimizer, batches and training trajectory. '
        'Compare model.pt (minimum Logloss) and model_ndcg.pt (maximum full NDCG).\n\n'
        'See metrics.json and run.json. These are selection-set results, not an independent '
        'holdout or ensemble improvement. Small first-feed sample is not representative.\n')


if __name__ == '__main__':
    main()
