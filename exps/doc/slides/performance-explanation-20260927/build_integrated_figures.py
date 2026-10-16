"""One data source for the integrated deck and manuscript; no extrapolated matrix values."""
from pathlib import Path
import json,csv,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent;ROOT=D.parents[2];A=D/'assets';A.mkdir(exist_ok=True)
PAPER=ROOT/'doc/paper/relconnector-background-method-revised-20260927'
DATA=json.loads((ROOT/'doc/targeted-batch-results-20260927.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
blue='#648CB6';orange='#C96843'
fig,axes=plt.subplots(1,2,figsize=(13,4.6))
export=[]
for ax,r in zip(axes,DATA['rows']):
 keys=['pipeline_s','ready_wait_s','wall_s']; labels=['Pipeline average','Ready wait / batch','Whole-run amortized']
 for y,key in enumerate(keys):
  m=r[key];b=m['baseline_ms_per_batch'];v=m['optimized_ms_per_batch']
  ax.plot([v,b],[y,y],color='#BFC7D1',lw=3,zorder=1)
  ax.scatter([b,v],[y,y],c=[blue,orange],s=75,zorder=3)
  for n,c in [(b,blue),(v,orange)]:ax.annotate(f'{n:,.1f}',(n,y),xytext=(0,12 if c==blue else -19),textcoords='offset points',ha='center',fontsize=11,color=c,weight='bold')
  ax.text(1.01,y,f"−{m['reduction_pct']:.2f}%",transform=ax.get_yaxis_transform(),va='center',color=orange,weight='bold')
  export.append({'dataset':r['dataset'],'metric':key,'before_ms_per_batch':b,'after_ms_per_batch':v,'reduction_pct':m['reduction_pct'],'batches_per_run':r['batches_per_run'],'repeats_per_variant':2})
 ax.set_yticks(range(3),labels);ax.set_ylim(2.6,-.7);ax.set_xlim(0,r['wall_s']['baseline_ms_per_batch']*1.10)
 ax.set_xlabel('ms / completed batch (lower is better)');ax.grid(axis='x',color='#EBEEF2');ax.set_axisbelow(True)
 ax.set_title(f"{r['dataset']}  |  {r['batches_per_run']} batches/run",loc='left',pad=18,weight='bold')
fig.legend([Line2D([0],[0],marker='o',ls='',color=blue),Line2D([0],[0],marker='o',ls='',color=orange)],['v2 before (matched control)','v2 optimized'],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,1.02))
fig.subplots_adjust(left=.16,right=.92,top=.79,bottom=.19,wspace=.9)
for ext in ['png','pdf','svg']:
 fig.savefig(A/f'v2-current-per-batch.{ext}',dpi=230,bbox_inches='tight')
 shutil.copy2(A/f'v2-current-per-batch.{ext}',PAPER/'figures'/f'fig03-v2-current-per-batch.{ext}')
plt.close(fig)
with (A/'v2-current-per-batch.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
# Keep the 10-dataset reference and latest targeted evidence visibly separate on one figure.
p=ROOT/'doc/figures/vanilla-v1-v2-20260927/final/_historical_plot.py'
s=p.read_text().replace('figsize=(16.5,7.5)','figsize=(16.5,8.4)').replace('bottom=.065','bottom=.20').replace("'v2 (multi process)'","'v2 (historical multi process)'")
s=s.replace("fig.text(.985,.944,'LOWER IS BETTER'","fig.text(.985,.944,'HISTORICAL CROSS-VERSION REFERENCE'")
insert="""
fig.text(.067,.09,'CURRENT MATCHED v2 EXPERIMENTS  |  pipeline ms / batch',fontsize=10,weight='bold',color='#252B33')
fig.text(.067,.051,CURRENT_LABEL,fontsize=11,color='#A64B2D')
fig.text(.067,.016,'Two repeats per variant; these short-run pipeline measurements do not replace the historical end-to-end bars or RSS.',fontsize=9,color='#697482')
"""
s=s.replace("for ext in ['png','pdf','svg']:fig.savefig",insert+"\nfor ext in ['png','pdf','svg']:fig.savefig")
# Execute the existing plot source with its original path so all source references remain valid.
label='     '.join(f"{r['dataset']}: {r['pipeline_s']['baseline_ms_per_batch']:,.1f} → {r['pipeline_s']['optimized_ms_per_batch']:,.1f} (−{r['pipeline_s']['reduction_pct']:.2f}%; {r['batches_per_run']} batches/run)" for r in DATA['rows'])
exec(compile(s,str(p),'exec'),{'__file__':str(p),'__name__':'__main__','CURRENT_LABEL':label})
# RSS: dot plots and explicit scope; no invented new memory measurement.
rows=list(csv.DictReader((ROOT/'doc/slides/bottleneck-review-20260927/rss-data.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(12.6,4.5),gridspec_kw={'width_ratios':[1,3.4]})
fields=['batch_reported_rss_gib','vanilla_reported_rss_gib','latest_reported_rss_gib','v2_main_process_rss_gib'];colors=['#A8B1BE','#737F8E',blue,orange]
for ax,subset,title in [(axs[0],rows[:1],'Amazon'),(axs[1],rows[1:],'Other datasets')]:
 for j,(key,c) in enumerate(zip(fields,colors)):
  for x,row in enumerate(subset):
   ax.scatter(x+(j-1.5)*.15,float(row[key]),s=36,facecolors='white' if j==3 else c,edgecolors=c,linewidths=1.4)
 ax.set_xticks(range(len(subset)),[r['dataset'].title() for r in subset]);ax.set_ylim(bottom=0);ax.grid(axis='y',color='#EBEEF2');ax.set_axisbelow(True);ax.set_title(title,loc='left',weight='bold')
axs[0].set_ylabel('Historical reported peak RSS (GiB)')
fig.legend([Line2D([0],[0],marker='o',ls='',color=c,markerfacecolor='white' if j==3 else c) for j,c in enumerate(colors)],['Batch','Vanilla','v1','v2 main process only'],ncol=4,loc='upper center',frameon=False)
fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.18,wspace=.24)
for ext in ['png','svg','pdf']:fig.savefig(A/f'rss-current-scope.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
print('Updated current paired evidence, historical comparison with latest-result strip, and RSS scope figure.')
