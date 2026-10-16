"""Absolute-value small multiples: 10 datasets, 2 rows x 5 columns."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator,FuncFormatter
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
SRC=ROOT/'doc/slides/bottleneck-review-20260927'
perf={x['dataset'].removeprefix('rel-'):x for x in json.loads((SRC/'performance-data.json').read_text())}
rss={x['dataset']:x for x in csv.DictReader((SRC/'rss-data.csv').open())}
order=['amazon','arxiv','avito','event','f1','hm','ratebeer','salt','stack','trial']
names=['Amazon','Arxiv','Avito','Event','F1','HM','Ratebeer','Salt','Stack','Trial']
colors=['#929AA5','#4B79AE','#218C80']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':.7,'axes.edgecolor':'#7A8590','xtick.color':'#47515D','ytick.color':'#47515D','text.color':'#25303C','axes.labelcolor':'#47515D','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.65,'savefig.facecolor':'white'})
fig,axs=plt.subplots(2,5,figsize=(18,7.5))
fig.subplots_adjust(left=.046,right=.963,top=.785,bottom=.20,wspace=.43,hspace=.48)
fig.suptitle('Vanilla vs. v1 vs. v2',fontsize=20,fontweight='bold',y=.978)
fig.text(.5,.922,'End-to-end time and reported peak RSS across ten datasets',ha='center',fontsize=12,color='#596574')
fig.legend(handles=[Patch(facecolor=c,edgecolor='white',label=l) for c,l in zip(colors,['Vanilla','v1 (single process)','v2'])],loc='upper center',bbox_to_anchor=(.5,.898),ncol=3,frameon=False,fontsize=11,columnspacing=2.8,handlelength=1.8)
export=[]
def fmt(v):
 return f'{v:.2f}' if v>=1 else f'{v:.3f}'
for i,(key,name,ax) in enumerate(zip(order,names,axs.flat)):
 p=perf[key];r=rss[key]
 times=[p['vanilla_s_per_batch'],p['latest_s_per_batch'],p['v2_s_per_batch']]
 mem=[float(r[k]) for k in ['vanilla_reported_rss_gib','latest_reported_rss_gib','v2_main_process_rss_gib']]
 right=ax.twinx();ax.set_xlim(-.65,6.65)
 ax.set_ylim(0,max(t for t in times if t is not None)*1.34);right.set_ylim(0,11.5)
 ax.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
 right.set_yticks([0,5,10]);right.tick_params(axis='y',labelsize=8.5,length=3,pad=3,colors='#697380')
 ax.tick_params(axis='y',labelsize=8.5,length=3,pad=3)
 for a in (ax,right):a.spines['top'].set_visible(False)
 ax.spines['right'].set_visible(False);right.spines['left'].set_visible(False)
 right.spines['bottom'].set_visible(False)
 ax.set_xticks([1,5],['Time','RSS']);ax.tick_params(axis='x',length=0,pad=8,labelsize=11)
 ax.text(0,1.017,'s / batch',transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5,color='#697380')
 ax.text(1,1.017,'GiB',transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,color='#697380')
 ax.set_title(f'({chr(97+i)}) {name}',fontsize=12,fontweight='bold',pad=23)
 ax.axvline(3,color='#DCE1E6',lw=.8,ls=(0,(3,3)),zorder=0)
 for j,(t,m,c) in enumerate(zip(times,mem,colors)):
  if t is None:
   ax.text(j,.06,'TO',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,fontweight='bold',color='#737C86')
   ax.plot([j-.22,j+.22],[.026,.026],transform=ax.get_xaxis_transform(),color=c,lw=1.4)
  else:
   approx=j==1 and p['latest_timeout_estimate']
   ax.bar(j,t,width=.68,color=c,edgecolor='#FFFFFF',linewidth=.7,hatch='..' if approx else None,zorder=3)
   label=('~' if approx else '')+fmt(t)
   lift=13 if j==1 and times[2] is not None and abs(t-times[2]) < max(v for v in times if v is not None)*.035 else 4
   ax.annotate(label,(j,t),xytext=(0,lift),textcoords='offset points',ha='center',va='bottom',fontsize=8.6)
  right.bar(4+j,m,width=.68,color=c,edgecolor='#FFFFFF' if j<2 else '#116E64',linewidth=.7,hatch='///' if j==2 else None,zorder=3,alpha=1 if j<2 else .80)
  right.annotate(f'{m:.2f}',(4+j,m),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8.6)
  export.append({'dataset':key,'version':['vanilla','v1','v2'][j],'time_s_per_batch':t,'time_status':'timeout' if t is None else 'partial-run throughput estimate' if j==1 and p['latest_timeout_estimate'] else 'completed','rss_gib':m,'rss_scope':'main process only' if j==2 else 'historical reported process RSS','rss_timeout_observation':(t is None or (j==1 and p['latest_timeout_estimate'])),'v2_source':r['v2_source'] if j==2 else ''})
fig.text(.046,.115,'Within each panel: Time uses the left axis; RSS uses the right axis. Linear scales; RSS scale shared across panels. Lower is better.',fontsize=10,color='#47515D')
fig.text(.046,.075,'TO: 4 h timeout (no completed-run time).  ~: v1 Ratebeer throughput from a timed-out run.  Hatched RSS: v2 main process only; child processes excluded.',fontsize=9.6,color='#47515D')
fig.text(.046,.036,'Historical runs and selected v2 reruns; cache states / epoch counts differ; v2 Ratebeer is capped. RSS for timed-out runs is the peak observed before timeout.',fontsize=9.4,color='#697380')
for ext in ['png','pdf','svg']:
 fig.savefig(OUT/f'vanilla-v1-v2-time-rss.{ext}',dpi=240)
fig.savefig(OUT/'preview.png',dpi=120)
with (OUT/'plotted-data.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
(OUT/'README.md').write_text('''# Vanilla / v1 / v2 十库对比图

v1 表示单进程优化版本，v2 表示多进程升级版本；历史原始字段名保留以便追溯。

2 行 × 5 列，每个数据集一个子图。每个子图包含 Time 与 RSS 两组柱，每组顺序为 Vanilla、v1、v2；版本配色在全部子图中一致。Time 使用左轴（端到端秒 / batch，包含准备），RSS 使用右轴（GiB）；均为从零开始的线性轴，RSS 轴十库共用相同范围，Time 轴各库独立。不同指标柱高不能直接比较。

Vanilla 五项超时显示 TO，不用 14400 秒或未经测量的完成耗时代替。v1 Ratebeer 为超时前吞吐估计，以 ~ 与点纹柱标识。超时 RSS 为截至终止的观测峰值。

v2 RSS 为 RUSAGE_SELF 主进程峰值，斜线标识，不含采样 / 组装 / manager，不能当作系统总内存或用此图断言整体内存节省。

耗时使用已有端到端秒 / batch 口径，历史一轮与 v2 两轮、冷热缓存状态不同；Event 是热初始化运行，Trial 使用已存在文本缓存；Ratebeer v2 每轮截断 10000 batch。不是严格同条件配对实验。无误差棒，因为没有独立重复实验。

数据来源：doc/slides/bottleneck-review-20260927/performance-data.json 和 rss-data.csv（历史报告与选定 v2 实测记录），未使用任何内存估算。

输出：PNG 预览/高分辨率，PDF/SVG 矢量，plotted-data.csv 逐项数值，plot_comparison.py 可复现脚本。未修改 PPT。
''')
print('Saved 10-panel comparison: PNG, PDF, SVG and data CSV.')
