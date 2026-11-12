"""仅核对输入形状，不生成分析结果。"""
import json
from .io import asset,integer
def summarize(v,root):
    integer(v['anticipation']);integer(v['bootstrap']['replicates'],1)
    raw=asset(root,v['panel']);rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)!=v['panel']['rows']:raise ValueError('行数不符')
    for r in rows:
        if not {'household','district','year','G','x','outcome'}<=set(r):raise ValueError('面板列缺失')
    return {'panel_rows':len(rows),'district_folds':len(v['folds']),'covariates':len(v['covariates'])}
