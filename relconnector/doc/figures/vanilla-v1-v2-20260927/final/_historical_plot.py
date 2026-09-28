from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator,FuncFormatter
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
SRC=ROOT/'doc/slides/bottleneck-review-20260927'
perf={r['dataset'].removeprefix('rel-'):r for r in json.loads((SRC/'performance-data.json').read_text())}
rss={r['dataset']:r for r in csv.DictReader((SRC/'rss-data.csv').open())}
keys=['amazon','arxiv','avito','event','f1','hm','ratebeer','salt','stack','trial']
names=['Amazon','Arxiv','Avito','Event','F1','HM','Ratebeer','Salt','Stack','Trial']
colors=['#AFB6C0','#648CB6','#C96843'];edge=['#7F8996','#426C95','#A64B2D']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#252B33','axes.labelcolor':'#444C56','axes.edgecolor':'#7B8490','axes.linewidth':.65,'xtick.color':'#414A56','ytick.color':'#65707D','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
fig=plt.figure(figsize=(16.5,7.5))
outer=fig.add_gridspec(2,5,left=.067,right=.985,bottom=.065,top=.885,wspace=.36,hspace=.40)
fig.legend(handles=[Patch(facecolor=c,edgecolor=e,linewidth=.6,label=l) for c,e,l in zip(colors,edge,['Vanilla','v1 (single process)','v2 (multi process)'])],loc='upper center',bbox_to_anchor=(.52,.989),ncol=3,frameon=False,fontsize=11.5,handlelength=1.6,columnspacing=3)
fig.text(.067,.944,'TIME  /  END-TO-END SECONDS PER BATCH',fontsize=8.8,color='#697482')
fig.text(.985,.944,'LOWER IS BETTER',fontsize=8.8,color='#697482',ha='right')
plotdata=[]
for i,(key,name) in enumerate(zip(keys,names)):
 sub=outer[i//5,i%5].subgridspec(2,1,height_ratios=[3.25,1],hspace=.16)
 ax=fig.add_subplot(sub[0]);mem=fig.add_subplot(sub[1],sharex=ax)
 p=perf[key];r=rss[key];ts=[p['vanilla_s_per_batch'],p['latest_s_per_batch'],p['v2_s_per_batch']]
 ms=[float(r[k]) for k in ['vanilla_reported_rss_gib','latest_reported_rss_gib','v2_main_process_rss_gib']]
 timed=[v is None or (j==1 and p['latest_timeout_estimate']) for j,v in enumerate(ts)]
 high=max(v for v,to in zip(ts,timed) if not to)*1.36
 ax.set_ylim(0,high);ax.set_xlim(-.6,2.6)
 ax.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3))
 ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
 ax.grid(axis='y',lw=.5,color='#E7EAEE',zorder=0)
 ax.tick_params(axis='y',length=2.5,labelsize=8.5,pad=3)
 ax.tick_params(axis='x',bottom=False,labelbottom=False)
 for a in (ax,mem):
  a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
 ax.set_title(f'({chr(97+i)})  {name}',loc='left',fontsize=12,fontweight='bold',pad=11)
 if i%5==0:ax.set_ylabel('Time (s / batch)',fontsize=10,labelpad=10)
 for j,(v,to,c,e) in enumerate(zip(ts,timed,colors,edge)):
  if to:
   ax.bar(j,high*.84,width=.37,color=c,edgecolor=e,linewidth=.6,zorder=3)
   for y in [.745,.79]:
    ax.plot([j-.235,j+.235],[(y-.017)*high,(y+.017)*high],lw=4.5,color='white',solid_capstyle='butt',zorder=5)
    ax.plot([j-.235,j+.235],[(y-.017)*high,(y+.017)*high],lw=.85,color=e,solid_capstyle='butt',zorder=6)
   ax.text(j,.90,'TO',transform=ax.get_xaxis_transform(),ha='center',fontsize=9.2,color=e,fontweight='bold')
  else:
   ax.bar(j,v,width=.37,color=c,edgecolor=e,linewidth=.65,zorder=3)
   label=f'{v:.2f}' if v>=1 else f'{v:.3f}'
   ax.annotate(label,(j,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9.4,fontweight='bold' if j==2 else 'normal',color=e if j==2 else '#343D48')
 # Secondary metric strip uses a common absolute scale and the same version positions.
 mem.set_facecolor('#F7F8FA');mem.set_ylim(0,12);mem.set_yticks([0,5,10])
 mem.tick_params(axis='y',length=0,labelsize=7.5,pad=4)
 mem.spines['left'].set_visible(False);mem.spines['bottom'].set_color('#D0D6DE')
 mem.set_xticks([0,1,2],['Vanilla','v1','v2']);mem.tick_params(axis='x',length=0,labelsize=9.5,pad=6)
 if i%5==0:mem.set_ylabel('RSS (GiB)',fontsize=9.5,labelpad=9)
 else:mem.tick_params(axis='y',labelleft=False)
 for j,(m,c,e) in enumerate(zip(ms,colors,edge)):
  mem.scatter(j,m,s=32,facecolors='white' if j==2 else c,edgecolors=e,linewidths=1.3,zorder=3)
  mem.annotate(f'{m:.2f}',(j,m),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8.0,color='#596573')
  plotdata.append({'dataset':key,'version':['vanilla','v1','v2'][j],'time_s_per_batch':ts[j],'time_plot_status':'timeout (symbolic broken bar)' if timed[j] else 'completed','rss_gib':m,'rss_scope':'main process only' if j==2 else 'historical process RSS'})
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'vanilla-v1-v2-time-rss.{ext}',dpi=260)
fig.savefig(OUT/'preview.png',dpi=140)
with (OUT/'plotted-data.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(plotdata[0]));w.writeheader();w.writerows(plotdata)
print('Exported research-informed revision 3.')
