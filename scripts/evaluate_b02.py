"""CPU evaluation of a B02 checkpoint; explicit val reference, no weight updates."""
import argparse
import json
from pathlib import Path
import time

from train_b02 import evaluate
import torch
from neuroforge.models.neural.history_mean import HistoryFeeds, make_ranker
from neuroforge.models.neural.progress import peak_ram_gib


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-feeds',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.max_feeds<0 or min(args.batch_size,args.threads)<1: parser.error('Invalid counts')
    start=time.perf_counter(); torch.set_num_threads(args.threads)
    ckpt=torch.load(args.model,map_location='cpu',weights_only=True)
    ds=HistoryFeeds(args.data/'val',args.max_feeds,mapping=ckpt['categories_mapping'])
    model=make_ranker(len(ds.mapping),use_history=ckpt['use_history'],**ckpt['model_params'])
    model.load_state_dict(ckpt['state_dict'])
    metrics,predictions=evaluate(model,ds,args.batch_size)
    metrics.update(seconds=time.perf_counter()-start,peak_ram_gib=peak_ram_gib(),
                   model=str(args.model),epoch=ckpt['epoch'],use_history=ckpt['use_history'])
    args.output.mkdir(parents=True)
    predictions.write_parquet(args.output/'predictions_val.parquet')
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__': main()
