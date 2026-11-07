"""Time-throttled console progress without changing data order or random state."""
import resource
import sys
import threading
import time

from torch.utils.data import DataLoader


def peak_ram_gib():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return size / (1024**3 if sys.platform == 'darwin' else 1024**2)


class ProgressLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.stage = 'train' if kwargs.get('shuffle', False) else 'validation'
        super().__init__(*args, **kwargs)
        self.pass_number = 0

    def __iter__(self):
        self.pass_number += 1
        start = last = time.monotonic()
        total = len(self)
        print(f'[{self.stage} pass {self.pass_number}] starting: {total} batches', flush=True)
        for index, batch in enumerate(super().__iter__(), 1):
            yield batch
            now = time.monotonic()
            if now - last >= 5 or index == total:
                elapsed = now - start
                eta = elapsed / index * (total - index)
                print(f'[{self.stage} pass {self.pass_number}] {index}/{total} '
                      f'({100*index/total:.1f}%) elapsed={elapsed:.0f}s ETA~{eta:.0f}s '
                      f'peak RAM={peak_ram_gib():.2f} GiB', flush=True)
                last = now


def install_progress(baseline):
    original = baseline.CandidateData

    class ProgressData(original):
        def __init__(self, split, max_feeds=0, categories_mapping=None):
            print(f'[loading] {split}, limit={max_feeds} whole feeds', flush=True)
            stop = threading.Event()

            def heartbeat():
                while not stop.wait(5):
                    print(f'[loading] still reading {split}; peak RAM={peak_ram_gib():.2f} GiB', flush=True)

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                super().__init__(split, max_feeds, categories_mapping)
            finally:
                stop.set()
                thread.join()
            print(f'[loaded] {len(self)} feeds, {len(self.keys)} candidates; '
                  f'peak RAM={peak_ram_gib():.2f} GiB', flush=True)

    baseline.CandidateData = ProgressData
    baseline.DataLoader = ProgressLoader
