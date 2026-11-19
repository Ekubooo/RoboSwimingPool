"""Exclusive CPU stage and GPU timeline partitions from an Nsight SQLite export."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import sqlite3


def partition(start, end, intervals, cpu=False):
    events = defaultdict(list)
    for ident, (a, b, label) in enumerate(intervals):
        if b <= start or a >= end:
            continue
        a, b = max(a, start), min(b, end)
        events[a].append((True, ident, label, b-a))
        events[b].append((False, ident, label, b-a))
    events[start]; events[end]
    active, result = {}, defaultdict(float)
    previous = start
    for t in sorted(events):
        if active:
            if cpu:
                label = min(active.values(), key=lambda item: item[1])[0]
            else:
                label = '+'.join(sorted(set(v[0] for v in active.values())))
        else:
            label = 'other_cpu' if cpu else 'no_traced_gpu_work'
        result[label] += (t-previous)/1e6
        for begin, ident, label, duration in events[t]:
            if begin:
                active[ident] = (label, duration)
            else:
                active.pop(ident, None)
        previous = t
    return dict(result)


def analyze(path):
    db = sqlite3.connect(path)
    names = dict(db.execute('select id,value from StringIds'))
    nvtx = [(a,b,text or names.get(tid,'')) for a,b,text,tid in
            db.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    frames = sorted((a,b,t) for a,b,t in nvtx if t.startswith('frame_'))
    stages = [(a,b,t) for a,b,t in nvtx if not t.startswith('frame_')]
    gpu = []
    kernels = list(db.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'))
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        gpu.extend((a,b,'CUDA') for a,b in db.execute(f'select start,end from {table}'))
    gpu.extend((a,b,'OpenGL') for a,b in db.execute('select start,end from OPENGL_WORKLOAD'))
    rows=[]
    for a,b,t in frames:
        rows.append(dict(frame=t,wall_ms=(b-a)/1e6,cpu_exclusive=partition(a,b,stages,True),
                         gpu_exclusive=partition(a,b,gpu)))
    means={}
    for field in ('cpu_exclusive','gpu_exclusive'):
        keys=set().union(*(row[field] for row in rows))
        means[field]={k:sum(row[field].get(k,0) for row in rows)/len(rows) for k in keys}
    first,last=frames[0][0],frames[-1][1]
    totals=defaultdict(float); counts=Counter()
    for a,b,n in kernels:
        if b>first and a<last:
            totals[names[n]]+=(min(b,last)-max(a,first))/1e6/len(rows)
            counts[names[n]]+=1
    apis={}
    for table in ('OPENGL_API','CUPTI_ACTIVITY_KIND_RUNTIME'):
        sums=defaultdict(float)
        for a,b,n in db.execute(f'select start,end,nameId from {table}'):
            if b>first and a<last:
                sums[names[n]]+=(min(b,last)-max(a,first))/1e6/len(rows)
        apis[table]=dict(sorted(sums.items(),key=lambda x:-x[1])[:15])
    return dict(frames=len(rows),mean_wall_ms=sum(r['wall_ms'] for r in rows)/len(rows),
                mean_partitions=means, representative_frame=sorted(rows,key=lambda r:r['wall_ms'])[len(rows)//2],
                top_cuda_kernels=[dict(name=k,ms_per_frame=v,calls=counts[k]) for k,v in sorted(totals.items(),key=lambda x:-x[1])[:20]],
                cpu_api_inclusive_ms=apis, samples=rows,
                note='Partitions clip to CPU NVTX frame intervals. GPU work may belong to adjacent submitted frames; no extra frame synchronization. CPU and GPU tables are alternate views, never additive. No traced GPU work is not proof of GPU idleness.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args(); result=analyze(a.sqlite)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','note')},indent=2))
