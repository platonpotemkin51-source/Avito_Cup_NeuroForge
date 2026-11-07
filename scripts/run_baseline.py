"""Configure upstream baselines without modifying original files."""
import importlib.util
import json
from pathlib import Path
import sys

def main():
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    kind = config['model']
    if kind not in ('catboost', 'neural'):
        raise ValueError('Unknown model')
    path = Path(__file__).resolve().parents[1] / 'baseline' / f'{kind}_baseline.py'
    spec = importlib.util.spec_from_file_location('upstream_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if kind == 'catboost':
        module.DATA_PARAMS.update(config.get('data', {}))
        module.MODEL_PARAMS.update(config.get('params', {}))
        if module.MODEL_PARAMS['task_type'] == 'CPU':
            for key in ('devices', 'gpu_ram_part'):
                module.MODEL_PARAMS.pop(key, None)
        prefix = []
    else:
        module.TRAIN_CONFIG.update(config.get('params', {}))
        device = config.get('device', 'cpu')
        if device == 'mps':
            if not module.torch.backends.mps.is_available():
                raise RuntimeError('Apple GPU (MPS) is unavailable in this environment')
            # Extend the organizer CLI in memory; baseline sources stay untouched.
            for option in module.main.params:
                if option.name == 'device':
                    option.type = module.click.Choice(['cpu', 'cuda', 'mps'])
        prefix = ['--device', device]
    module.main(args=prefix + sys.argv[2:], standalone_mode=False)

if __name__ == '__main__':
    main()
