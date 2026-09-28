from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D=Path(__file__).resolve().parent; F=D/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#172b45','axes.edgecolor':'#9aabba','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
blue='#4379a8'; teal='#24958c'; gray='#a9b5c3'; orange='#d39446'
def save(fig,name):
 for ext in ['png','pdf','svg']: fig.savefig(F/f'{name}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
def clean(ax):
 ax.set_axisbelow(True);ax.grid(axis='y',color='#e7edf2',linewidth=.7)
def labels(ax,bars,fmt='{:.1f}',offset=3):
 ax.bar_label(bars,labels=[fmt.format(p.get_height()) for p in bars],padding=offset,fontsize=9)
# 1 historical memory
names=['Amazon','Arxiv','Avito','Event','F1','HM','Ratebeer','Salt','Stack','Trial']
b=[103.82,7.33,17.39,31.93,2.29,14.35,20.29,17.73,19.50,22.04]
v=[4.23,4.29,4.26,3.91,2.44,3.89,3.86,5.15,3.67,2.92]
r=[7.23,8.15,4.51,4.24,2.57,5.26,8.74,9.29,8.16,4.50]
fig,ax=plt.subplots(figsize=(10.8,3.7));x=np.arange(10)
for off,vals,color,label in [(-.24,b,gray,'Full materialization (Batch)'),(0,v,blue,'On-demand (Vanilla)'),(.24,r,teal,'On-demand + reuse (v1)')]:ax.bar(x+off,vals,.21,color=color,label=label)
ax.set_yscale('log');ax.set_ylim(1,180);ax.set_yticks([1,2,4,8,16,32,64,128],labels=['1','2','4','8','16','32','64','128']);ax.set_xticks(x,names);ax.set_ylabel('Historical peak RSS (GiB)');clean(ax)
ax.axhline(32,color=orange,lw=1,ls='--');ax.text(9.4,35,'32 GiB reference',ha='right',fontsize=9,color=orange)
ax.annotate('103.82',(-.24,103.82),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3,frameon=False,fontsize=9);save(fig,'fig01-memory')
# 2 architecture
fig,ax=plt.subplots(figsize=(13.5,6.4));ax.set_xlim(0,13.5);ax.set_ylim(0,6.4);ax.axis('off')
def box(x,y,w,h,title,body,fc='#edf4fa',ec=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.10',facecolor=fc,edgecolor=ec,lw=1.2))
 ax.text(x+w/2,y+h-.3,title,ha='center',va='center',weight='bold',fontsize=9.5)
 ax.text(x+w/2,y+h/2-.12,body,ha='center',va='center',fontsize=8.5,linespacing=1.65)
def arrow(a,b,label=None,dy=.12):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#60758a'))
 if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+dy,label,ha='center',fontsize=8.5,bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
ax.text(.15,6.12,'RelConnector: separate reusable preparation from demand-driven execution',weight='bold',fontsize=15)
box(.2,4.05,2.0,1.45,'Relational database','Rows · column types\nPrimary / foreign keys\nTimestamps · features',fc='#f4f6f8',ec=gray)
box(3,4.05,4.6,1.45,'1  Offline preparation','Time-filtered schema statistics + graph adjacency\nBudgeted exact statistics · versioned artifacts')
box(8.4,4.05,4.65,1.45,'Reusable artifacts','Schema / statistics + topology\nTopology remains resident in host memory')
arrow((2.24,4.78),(2.96,4.78));arrow((7.64,4.78),(8.36,4.78))
ax.axhline(3.62,xmin=.015,xmax=.97,color='#bac7d3',ls='--',lw=1)
ax.text(.23,3.76,'OFFLINE  /  reusable across runs',fontsize=9,color='#61758a')
ax.text(.23,3.36,'ONLINE  /  bounded windows and queues',fontsize=9,color='#61758a')
box(.2,1.48,2.4,1.43,'2  Seed sampling','One sampling process\nTemporal neighborhoods')
box(3.25,1.48,2.2,1.43,'Plan window','IDs · edges · timestamps\nBatch / byte limits',fc='#f4f6f8',ec=gray)
box(6.15,1.48,3.1,1.43,'3  Parallel assembly','Fetch rows → static encoding\nRestore batch layout\nMultiple assembly workers')
box(10.25,1.48,2.8,1.43,'4  GPU model training','Trainable encoders + GNN\nLoss + backward + optimizer',fc='#f1f3f5',ec=gray)
arrow((2.64,2.2),(3.21,2.2));arrow((5.49,2.2),(6.11,2.2));arrow((9.29,2.2),(10.21,2.2),'Ready batches')
arrow((9,4.02),(1.4,2.97),'Topology + time indexes',dy=.02)
arrow((11,4.02),(7.7,2.97),'Schema',dy=.03)
box(3.25,.15,6,0.85,'Cost-aware feature supply and reuse','SQL blocks | shared/local encoded rows | text slots + compact disk shards',fc='#edf8f5',ec=teal)
arrow((7.7,1.05),(7.7,1.44))
ax.text(1.35,.82,'Seeds and prediction times',ha='center',fontsize=9);arrow((1.4,1.0),(1.4,1.44))
ax.text(11.65,.72,'Model computation is unchanged\nand outside this work\'s optimization scope.',ha='center',fontsize=9,color='#65758a')
save(fig,'fig02-architecture')

import runpy
runpy.run_path(str(D.parents[2]/'doc/slides/performance-explanation-20260927/build_integrated_figures.py'),run_name='__main__')
print('Built 3 integrated figures; current paired data come from targeted-batch-results-20260927.json.')
