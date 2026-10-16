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
with (D/'figure-data.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['figure','dataset','metric','variant','value','unit','source'])
 for name,bb,vv,rr in zip(names,b,v,r):
  for variant,val in [('Batch',bb),('Vanilla',vv),('v1',rr)]:w.writerow(['1',name,'observed peak RSS',variant,val,'GiB','doc/gpu-feature-optimization-experiment.md'])
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
box(3.25,.15,6,0.85,'Feature supply and reuse','SQL blocks  |  encoded rows  |  shared / persistent text vectors',fc='#edf8f5',ec=teal)
arrow((7.7,1.05),(7.7,1.44))
ax.text(1.35,.82,'Seeds and prediction times',ha='center',fontsize=9);arrow((1.4,1.0),(1.4,1.44))
ax.text(11.65,.72,'Model computation is unchanged\nand outside this work\'s optimization scope.',ha='center',fontsize=9,color='#65758a')
save(fig,'fig02-architecture')
# 3 initialization
fig,axes=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={'width_ratios':[1,1.25]})
ax=axes[0];x=np.arange(2);bottom=np.zeros(2)
for vals,color,label in [([1895.62,335.66],blue,'Schema statistics'),([267.95,269.92],teal,'Topology'),([.43,.42],gray,'Other')]:ax.bar(x,vals,.44,bottom=bottom,color=color,label=label);bottom+=vals
ax.set_xticks(x,['Before','After']);ax.set_ylabel('Cold preparation time (s)');ax.set_ylim(0,2600);clean(ax)
for xx,val in zip(x,[2164,606.01]):ax.text(xx,val+55,f'{val:,.2f}',ha='center')
ax.set_title('(a) Event: 3.57× faster preparation',fontsize=11);ax.legend(frameon=False,fontsize=9)
ax=axes[1];ds=['Amazon','Arxiv','Stack','HM','Salt','Trial','Avito'];vals=[97.3,89,86.5,65.6,55.3,60.4,50.6]
ax.barh(ds[::-1],vals[::-1],height=.57,color=blue);ax.set_xlim(0,110);ax.set_xlabel('Time waiting for a ready batch (%)');ax.set_title('(b) Historical training-side waiting',fontsize=11);ax.grid(axis='x',color='#e7edf2');ax.set_axisbelow(True)
for y,val in enumerate(vals[::-1]):ax.text(val+1,y,f'{val:.1f}',va='center',fontsize=9)
fig.tight_layout(w_pad=3);save(fig,'fig03-bottlenecks')
# 4 sample demand
fig,axes=plt.subplots(1,2,figsize=(9,3.4));ax=axes[0];bars=ax.bar(['Salt','Ratebeer'],[2.114,1.105],.4,color=[teal,blue]);labels(ax,bars,'{:.3f}');ax.axhline(1,color=gray,ls='--');ax.set_ylim(0,2.65);ax.set_ylabel('Requested rows / unique rows');ax.set_title('(a) Duplication varies across workloads',fontsize=11);clean(ax)
ax=axes[1];bars=ax.bar(['Requested','Unique','Read via SQL'],[122151,110528,88253],.42,color=[gray,blue,teal]);labels(ax,bars,'{:,.0f}');ax.set_ylim(0,153000);ax.set_ylabel('Rows per batch');ax.set_title('(b) Ratebeer: substantial residual demand',fontsize=11);clean(ax);fig.tight_layout(w_pad=3);save(fig,'fig04-demand')
# 5 salt v1 independent times
fig,axes=plt.subplots(1,3,figsize=(10.3,3.3))
for ax,title,vals in zip(axes,['End-to-end wall time','Feature fetch service time','Assembly service time'],[[14132.8,2832.6],[1169.6,261.8],[12917.6,2525.2]]):
 bars=ax.bar(['Vanilla','v1'],vals,.42,color=[gray,teal]);labels(ax,bars,'{:,.1f}');ax.set_ylim(0,max(vals)*1.23);ax.set_ylabel('Seconds');ax.set_title(title,fontsize=10);clean(ax)
fig.tight_layout(w_pad=2);save(fig,'fig05-salt-reuse')
# 6 costly hits
fig,axes=plt.subplots(1,3,figsize=(11.7,3.5))
for ax,title,vals,unit in [(axes[0],'(a) Salt: full two-epoch run',[18260.23,5208.23],'Wall time (s)'),(axes[1],'(b) Salt: shared text lookup',[30532,802.46],'Accumulated worker time (s)')]:
 bars=ax.bar(['Original v2','Revised v2'],vals,.42,color=[gray,teal]);labels(ax,bars,'{:,.0f}');ax.set_ylim(0,max(vals)*1.25);ax.set_ylabel(unit);ax.set_title(title,fontsize=10);clean(ax)
ax=axes[2];bars=ax.bar(['Shared\nfull blocks','Shared\nrequested rows','Local\nworker cache'],[206,183.5,157.7],.45,color=[gray,blue,teal]);labels(ax,bars);ax.set_ylim(0,255);ax.set_ylabel('Wall time (s)');ax.set_title('(c) HM: equal 2 GiB cache budget',fontsize=10);ax.tick_params(axis='x',labelsize=8);clean(ax);fig.tight_layout(w_pad=2);save(fig,'fig06-hit-cost')
print('Built 6 figures in PNG, PDF and SVG')

# Export the remaining plotted measurements with units and provenance.
with (D/'figure-data.csv').open('a') as f:
 w=csv.writer(f)
 for variant,vals in [('Before',[1895.62,267.95,.43]),('After',[335.66,269.92,.42])]:
  for metric,val in zip(['schema','topology','other'],vals):w.writerow(['3a','Event',metric,variant,val,'s','doc/v2-initialization-optimization-20260927.md'])
 for ds,val in zip(['Amazon','Arxiv','Stack','HM','Salt','Trial','Avito'],[97.3,89,86.5,65.6,55.3,60.4,50.6]):w.writerow(['3b',ds,'ready wait / train wall','historical diagnostic',val,'percent','doc/bottleneck-analysis-and-optimization.md'])
 for ds,val in [('Salt',2.114),('Ratebeer',1.105)]:w.writerow(['4a',ds,'requested / unique','historical',val,'ratio','doc/gpu-feature-optimization-experiment.md'])
 for metric,val in zip(['requested','unique','SQL read'],[122151,110528,88253]):w.writerow(['4b','Ratebeer',metric,'historical diagnostic',val,'rows/batch','doc/bottleneck-analysis-and-optimization.md'])
 for metric,vals in [('wall',[14132.8,2832.6]),('fetch service',[1169.6,261.8]),('assemble service',[12917.6,2525.2])]:
  for variant,val in zip(['Vanilla','v1'],vals):w.writerow(['5','Salt',metric,variant,val,'s','doc/gpu-feature-optimization-experiment.md'])
 for metric,vals,unit in [('wall',[18260.23,5208.23],'s'),('shared lookup',[30532,802.46],'worker-s')]:
  for variant,val in zip(['Original v2','Revised v2'],vals):w.writerow(['6','Salt',metric,variant,val,unit,'doc/v2-salt-regression-20260926.md'])
 for variant,val in zip(['Shared full blocks','Shared requested rows','Local worker cache'],[206,183.5,157.7]):w.writerow(['6c','HM','wall',variant,val,'s','doc/v2-hm-regression-20260925.md'])
