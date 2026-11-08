"""Audit a matched B02 pair and compare frozen predictions by feed/user."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import polars as pl
from neuroforge.metrics import ndcg


def compare(data, control, history, output):
    if output.exists(): raise FileExistsError(output)
    spec=importlib.util.spec_from_file_location('b02_uncertainty',ROOT/'scripts/compare_b01.py')
    uncertainty=importlib.util.module_from_spec(spec);spec.loader.exec_module(uncertainty)
    source=pl.read_parquet(sorted((data/'val/impressions').glob('*.parquet')),
                           columns=['feed_id','user_id','item_id','target','history__item_id'])
    meta=source.select('feed_id','user_id',pl.col('history__item_id').list.len().alias('history_length'))
    labels=source.select('feed_id','item_id','target').explode(['item_id','target'])
    runs={};frames={};manifests={}
    for variant,folder in [('control',control),('history',history)]:
        cfg=json.loads((folder/'config.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        epochs=json.loads((folder/'history.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        if cfg['variant']!=variant or len(epochs)!=cfg['epochs'] or epochs[-1]['epoch']!=cfg['epochs']:
            raise ValueError('Wrong variant or incomplete training')
        if max(epochs,key=lambda e:e['ndcg'])['epoch']!=metrics['epoch']:
            raise ValueError('Checkpoint does not match selection rule')
        pred=pl.read_parquet(folder/'predictions_val.parquet')
        if len(pred)!=len(labels) or pred.schema['score']!=pl.Float32 or not pred['score'].is_finite().all():
            raise ValueError('Prediction count/dtype/score mismatch')
        joined=labels.join(pred,on=['feed_id','item_id'],how='left',validate='1:1')
        if joined['score'].null_count(): raise ValueError('Missing reference keys')
        rows=[]
        for key,g in joined.group_by('feed_id',maintain_order=True):
            rows.append({'feed_id':key[0],'ndcg':ndcg(g['target'].to_list(),g['score'].to_list())})
        frames[variant]=pl.DataFrame(rows).cast({'feed_id':pl.Int32})
        if abs(frames[variant]['ndcg'].mean()-metrics['ndcg'])>1e-10:
            raise ValueError('Saved metric differs from prediction metric')
        manifests[variant]=manifest
        runs[variant]={'config':cfg,'metrics':metrics,'history':epochs,
                       **{k:manifest[k] for k in ['seconds','peak_ram_gib','parameters']}}
    a=runs['control']['config'].copy();b=runs['history']['config'].copy()
    a.pop('variant');b.pop('variant')
    if a!=b or runs['control']['parameters']!=runs['history']['parameters']:
        raise ValueError('Conditions/parameter count differ')
    for k in ['parts','source_sha256','python','torch']:
        if manifests['control'][k]!=manifests['history'][k]: raise ValueError('Run provenance differs: '+k)
    paired=frames['history'].join(frames['control'].rename({'ndcg':'control_ndcg'}),on='feed_id').join(meta,on='feed_id').with_columns(
        (pl.col('ndcg')-pl.col('control_ndcg')).alias('difference'))
    result={'runs':runs,'feeds':len(paired),'candidates':len(labels),'difference':paired['difference'].mean(),
            'paired_user_bootstrap_95pct':uncertainty.cluster_interval(paired),
            'slice_by_history':paired.with_columns(pl.when(pl.col('history_length')==0).then(pl.lit('empty')).when(pl.col('history_length')<=32).then(pl.lit('1-32')).otherwise(pl.lit('33-256')).alias('slice')).group_by('slice').agg(pl.len().alias('feeds'),pl.col('difference').mean()).sort('slice').to_dicts(),
            'limitations':'One training seed per pair; same full val used for selection and evaluation; bootstrap conditional on fixed models.'}
    output.mkdir(parents=True)
    paired.write_parquet(output/'per_feed_comparison.parquet')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['data','control','history','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();compare(args.data,args.control,args.history,args.output)


if __name__=='__main__':main()
