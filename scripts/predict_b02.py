"""Stream B02 checkpoint scores for test into one submission Parquet."""
import argparse
import json
from pathlib import Path
import tempfile

from train_b02 import forward
import polars as pl
import torch
from neuroforge.models.neural.history_mean import HistoryFeeds, collate_history, make_ranker
from neuroforge.models.neural.progress import ProgressLoader


def predict(data, checkpoint, output, batch_size=2, threads=4):
    if output.exists(): raise FileExistsError(output)
    if min(batch_size, threads)<1: raise ValueError('Positive batch size/threads required')
    torch.set_num_threads(threads)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dataset=HistoryFeeds(data/'test',mapping=saved['categories_mapping'],
                         include_history=saved['use_history'],release_pages=True)
    model=make_ranker(len(dataset.mapping),use_history=saved['use_history'],**saved['model_params'])
    model.load_state_dict(saved['state_dict']);model.eval()
    output.parent.mkdir(parents=True,exist_ok=True)
    feeds=count=0
    with tempfile.TemporaryDirectory(prefix='predict-parts-',dir=output.parent) as directory:
        directory=Path(directory)
        with torch.inference_mode():
            loader=ProgressLoader(dataset,batch_size=batch_size,collate_fn=collate_history)
            loader.stage='test'
            for i,batch in enumerate(loader):
                values=forward(model,batch)[~batch['padding']].float().cpu().numpy()
                frame=pl.DataFrame({'feed_id':[k[0] for k in batch['keys']],
                                    'item_id':[k[1] for k in batch['keys']], 'score':values},
                                   schema={'feed_id':pl.Int32,'item_id':pl.Int32,'score':pl.Float32})
                if not frame['score'].is_finite().all(): raise ValueError('Nonfinite scores')
                frame.write_parquet(directory/f'part-{i:08d}.parquet')
                feeds+=len(batch['users']);count+=len(frame)
        if feeds!=len(dataset): raise ValueError('Feed coverage mismatch')
        temporary=directory/'submission.tmp'
        pl.scan_parquet(str(directory/'part-*.parquet')).sink_parquet(temporary,maintain_order=True)
        temporary.replace(output)
    print(json.dumps({'feeds':feeds,'candidates':count,'output':str(output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data','model','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();predict(args.data,args.model,args.output,args.batch_size,args.threads)


if __name__=='__main__': main()
