"""Rebuild the editable narrative deck and reproducible scientific charts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]
ASSETS=OUT/'assets';ASSETS.mkdir(exist_ok=True)
ROWS=json.loads((BASE/'benchmarks/v2-performance-review-20260927/comparison.json').read_text())
DATA={r['dataset'].removeprefix('rel-'):r for r in ROWS}
ORDER=['amazon','arxiv','avito','event','f1','hm','ratebeer','salt','stack','trial']
NAMES={'amazon':'Amazon','arxiv':'Arxiv','avito':'Avito','event':'Event','f1':'F1','hm':'HM','ratebeer':'Ratebeer','salt':'Salt','stack':'Stack','trial':'Trial'}
INK='172B3A';BLUE='3567A8';TEAL='008879';GREY='88929C';LIGHT='EDF2F5';WHITE='FFFFFF';MUTED='586B79';ORANGE='B36A24'
FONT='Noto Sans CJK SC'
plt.rcParams.update({'font.family':FONT,'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,'savefig.facecolor':'white'})
fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
from matplotlib import font_manager
font_manager.fontManager.addfont(fp.get_file())
plt.rcParams['font.family']=fp.get_name()

# Performance charts exclude Batch completely. Timeout bars are lower bounds.
def performance(keys,name):
 fig,ax=plt.subplots(figsize=(12.1,4.55));ys=np.arange(len(keys))*1.18
 colors=['#'+GREY,'#'+BLUE,'#'+TEAL]
 for si,label in enumerate(['Vanilla','v1','v2']):
  vals=[]
  for k in keys:
   r=DATA[k]
   vals.append((r['vanilla_s_per_batch'] if r['vanilla_s_per_batch'] is not None else 14400/r['historical_epoch_batches']) if si==0 else r['latest_s_per_batch'] if si==1 else r['v2_s_per_batch'])
  bars=ax.barh(ys+(si-1)*.25,np.array(vals)-.035,left=.035,height=.215,color=colors[si],label=label,zorder=3)
  for i,(k,val,bar) in enumerate(zip(keys,vals,bars)):
   timeout=si==0 and DATA[k]['vanilla_total_s'] is None
   estimated=si==1 and DATA[k]['latest_timeout_estimate']
   warm=si==2 and k in ['event','trial']
   if timeout:
    bar.set_facecolor('#E0E4E8');bar.set_edgecolor('#'+GREY);bar.set_hatch('///')
   prefix='>' if timeout else '≈' if estimated else ''
   suffix='†' if warm else ''
   ax.text(val*1.055,ys[i]+(si-1)*.25,f'{prefix}{val:.3f}{suffix}',va='center',fontsize=11,color=colors[si])
 ax.set_yticks(ys,[NAMES[k] for k in keys],fontsize=14,fontweight='bold');ax.invert_yaxis()
 ax.set_xscale('log');ax.set_xlim(.035,65)
 ticks=[.05,.1,.3,1,3,10,30];ax.set_xticks(ticks,[str(t) for t in ticks]);ax.minorticks_off()
 ax.set_xlabel('端到端秒 / batch（对数轴，越低越好）',fontsize=12,labelpad=8)
 ax.grid(axis='x',color='#E4E8ED',zorder=0);ax.tick_params(axis='y',length=0,pad=10)
 ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,fontsize=12)
 fig.subplots_adjust(left=.10,right=.94,top=.88,bottom=.14)
 fig.savefig(ASSETS/f'{name}.png',dpi=200);fig.savefig(ASSETS/f'{name}.svg');plt.close(fig)
performance(['amazon','arxiv','avito','hm','ratebeer'],'performance_a')
performance(['event','f1','salt','stack','trial'],'performance_b')

RSS={
'amazon':[103.82,4.23,7.23],'arxiv':[7.33,4.29,8.15],'avito':[17.39,4.26,4.51],
'event':[31.93,3.91,4.24],'f1':[2.29,2.44,2.57],'hm':[14.35,3.89,5.26],
'ratebeer':[20.29,3.86,8.74],'salt':[17.73,5.15,9.29],'stack':[19.50,3.67,8.16],
 'trial':[22.04,2.92,4.50]}
for k in ORDER:
 path,line=DATA[k]['source'].rsplit(':',1)
 r=json.loads((BASE/path).read_text().splitlines()[int(line)-1])
 RSS[k].append(r['peak_rss_mb']/1024)
with (OUT/'rss-data.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['dataset','batch_reported_rss_gib','vanilla_reported_rss_gib','latest_reported_rss_gib','v2_main_process_rss_gib','v2_source'])
 for k in ORDER:w.writerow([k,*RSS[k],DATA[k]['source']])
fig,ax=plt.subplots(figsize=(12.1,4.5));x=np.arange(10)
for j,(label,color) in enumerate(zip(['Batch','Vanilla','v1','v2 主进程（不含子进程）'],['#'+ORANGE,'#'+GREY,'#'+BLUE,'#'+TEAL])):
 vals=[RSS[k][j] for k in ORDER]
 bars=ax.bar(x+(j-1.5)*.19,vals,width=.17,label=label,color=color,zorder=3)
 if j==3:
  for b in bars:b.set_facecolor('white');b.set_edgecolor(color);b.set_hatch('////');b.set_linewidth(1.3)
ax.set_yscale('log',base=2);ax.set_ylim(1,150);ax.set_yticks([1,2,4,8,16,32,64,128],[1,2,4,8,16,32,64,128]);ax.minorticks_off()
ax.set_xticks(x,[NAMES[k] for k in ORDER],fontsize=12)
ax.set_ylabel('峰值 RSS / GiB（对数轴）',fontsize=12)
ax.grid(axis='y',color='#E4E8ED',zorder=0);ax.tick_params(axis='x',length=0)
ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(-.025,1.02),frameon=False,fontsize=11)
ax.annotate('103.82',xy=(-.285,103.82),xytext=(-.285,125),ha='center',fontsize=11,color='#'+ORANGE)
ax.annotate('31.93',xy=(3-.285,31.93),xytext=(3-.285,39),ha='center',fontsize=11,color='#'+ORANGE)
fig.subplots_adjust(left=.07,right=.99,top=.86,bottom=.14)
fig.savefig(ASSETS/'rss.png',dpi=200);fig.savefig(ASSETS/'rss.svg');plt.close(fig)

R=Presentation();R.slide_width=Inches(13.333);R.slide_height=Inches(7.5)
NOTES=[]
def color(c):return RGBColor.from_string(c)
def text(s,x,y,w,h,value,size=20,bold=False,c=INK,align=None):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(str(value).split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(c);p.space_after=Pt(7);p.line_spacing=1.08
  if align is not None:p.alignment=align
  for run in p.runs:
   ea=OxmlElement('a:ea');ea.set('typeface',FONT);run._r.get_or_add_rPr().append(ea)
 return sh

def box(s,x,y,w,h,fill=LIGHT,stroke=None):
 sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=color(fill)
 if stroke:sh.line.color.rgb=color(stroke)
 else:sh.line.fill.background()
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 return sh

def line(s,x1,y1,x2,y2,c=GREY,arrow=False):
 sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.width=Pt(1.5);sh.line.color.rgb=color(c)
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 if arrow:
  e=OxmlElement('a:tailEnd');e.set('type','triangle');sh._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln').append(e)
 return sh

def node(s,x,y,w,h,label,sub='',fill=LIGHT):
 box(s,x,y,w,h,fill);text(s,x+.13,y+.12,w-.26,.42,label,20,True)
 if sub:text(s,x+.13,y+.65,w-.26,h-.73,sub,16,c=MUTED)

def slide(section,title,sub='',source='2026-09-27 · 各任务最近有效成绩'):
 s=R.slides.add_slide(R.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(WHITE)
 text(s,.55,.24,11.7,.25,section,11,True,TEAL)
 text(s,.55,.70,12.2,.70,title,29,True)
 if sub:text(s,.57,1.45,12.1,.50,sub,17,c=MUTED)
 line(s,.55,7.12,12.78,7.12,c='D6DFE5');text(s,.57,7.22,11.6,.18,source,8.5,c=MUTED)
 text(s,12.23,7.17,.53,.28,f'{len(R.slides):02}',11,c=MUTED,align=PP_ALIGN.RIGHT)
 return s

def takeaway(s,value):
 box(s,.55,6.40,12.23,.53,fill='E8F4F1');text(s,.73,6.49,11.86,.34,value,17,True,TEAL)

def note(s,value,refs):
 full=value+'\n\n证据与口径：\n'+'\n'.join(refs)
 s.notes_slide.notes_text_frame.text=full
 title=next(sh.text for sh in s.shapes if sh.has_text_frame and sh.top==Inches(.70))
 NOTES.append({'page':len(R.slides),'title':title,'notes':full})

def table(s,headers,rows,widths,y=2.05,rowh=.39,fontsize=16):
 shp=s.shapes.add_table(len(rows)+1,len(headers),Inches(.6),Inches(y),Inches(sum(widths)),Inches(rowh*(len(rows)+1)));tab=shp.table
 for j,w in enumerate(widths):tab.columns[j].width=Inches(w)
 for i,row in enumerate([headers]+rows):
  tab.rows[i].height=Inches(rowh)
  for j,val in enumerate(row):
   cell=tab.cell(i,j);cell.text=str(val);cell.margin_left=Inches(.10);cell.margin_right=Inches(.07);cell.margin_top=Inches(.055);cell.margin_bottom=Inches(.025);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
   cell.fill.solid();cell.fill.fore_color.rgb=color(INK if i==0 else WHITE if i%2==0 else LIGHT)
   for p in cell.text_frame.paragraphs:
    p.font.name=FONT;p.font.size=Pt(fontsize);p.font.bold=i==0;p.font.color.rgb=color(WHITE if i==0 else INK);p.space_after=Pt(0)
 return shp

def metric(s,x,y,w,big,label):
 text(s,x,y,w,.65,big,36,True,TEAL);text(s,x,y+.82,w,.9,label,18,c=MUTED)

def beforeafter(s,x,y,w,title,a,b,labels=('优化前','优化后'),unit='秒'):
 text(s,x,y,w,.4,title,20,True)
 for i,(v,lab,c) in enumerate(zip([a,b],labels,[GREY,TEAL])):
  yy=y+.73+i*.72;text(s,x,yy,1.2,.3,lab,15,c=MUTED)
  bw=(w-2.25)*v/max(a,b);box(s,x+1.3,yy+.02,bw,.29,c)
  text(s,x+1.38+bw,yy-.03,1.15,.34,f'{v:,.1f}',15,True,c)
 text(s,x,y+2.2,w,.35,f'{a/b:.2f}×  ·  {unit}',18,True,TEAL)

# 1
s=slide('REL CONNECTOR / 优化复盘','从特征供给瓶颈，到自适应在线训练','核心问题：特征不能全量驻留时，怎样让 GPU 持续获得训练数据？')
for i,(label,sub) in enumerate([('Vanilla','按需读取\n避免全量物化'),('v1','单进程优化\n去重与复用'),('v2','多进程升级\n采样 / 组装 / 训练分离')]):
 node(s,.65+i*4.15,2.30,3.68,1.6,label,sub)
 if i<2:line(s,4.38+i*4.15,3.10,4.72+i*4.15,3.10,TEAL,True)
text(s,.7,4.58,11.8,1.1,'汇报顺序\n问题 → v1 → 新瓶颈 → v2 架构与算法 → 实验 → 耗时图 → RSS 图',21)
takeaway(s,'优化对象不是数据集名称，而是重复率、访问成本、工作集大小和资源压力。')
note(s,'重组旧稿，更新至十任务完成及最新Event/Trial复测。所有历史对比均保留缓存与版本差异。性能图不含Batch；内存图按用户要求包含Batch。', ['doc/v2-dataset-performance-review-20260927.md'])
#2
s=slide('01 / VANILLA 的问题','按需读取省下内存，却把特征生产放进训练循环','Vanilla 已有基础 SQL / raw cache；问题是重复生产和串行供给仍然昂贵。')
for i,(a,b) in enumerate([('采样节点','同一节点可能重复出现'),('SQL + 解码','重复访问与数据转换'),('文本 + Frame','重复编码与组装'),('GPU 更新','等待 ready 数据')]):
 node(s,.65+i*3.15,2.15,2.7,1.55,a,b)
 if i<3:line(s,3.40+i*3.15,2.92,3.73+i*3.15,2.92,TEAL,True)
text(s,.7,4.10,5.65,1.6,'问题 A｜重复工作\n同一节点、同一文本被多次处理；\n原始行缓存不等于最终特征复用。',21)
text(s,6.9,4.10,5.55,1.6,'问题 B｜供给不足\nSQL、编码和组装阻塞训练；\n端到端耗时还包含首次初始化。',21)
takeaway(s,'先减少每个 batch 的重复工作，再提高特征生产的并行度。')
note(s,'历史Vanilla五项在4小时超时。不要把Vanilla描述为完全没有优化；旧SQL修复已回补到当前代码，历史成绩代表当时快照。', ['doc/gpu-feature-optimization-experiment.md','doc/optimization-evolution-report-20260926.md §2–3'])
#3
s=slide('02 / v1 如何优化','把重复请求合并，只对缺失部分生产特征','三层复用：节点请求、文本向量、已编码特征。')
table(s,['改动','省掉什么','必须保留什么'],[
['节点去重 + 跨 batch 窗口','重复 SQL / 解码 / 编码','恢复每个 batch 的节点位置和顺序'],
['稀疏 ID 查询 / 稠密块读取','无效扫描和过量取数','相同请求行与时间约束'],
['raw → text → encoded 缓存','逐步省掉后续转换工作','仅缓存静态特征、冻结文本向量'],
['direct 文本路径 + 有界线程流水线','通用调度开销，重叠 SQL 与编码','兼容回退；限制队列与工作集'],
], [3.65,4.05,4.40],rowh=.73,fontsize=18)
takeaway(s,'v1 的主要收益来自“少读、少转、少编码”，不改变训练输入的数量。')
note(s,'去重只减少特征生产，重复训练位置通过inverse恢复。文本快路径保留GloVe计算语义；不兼容时回退。初始化持久化缓存并非v2独有。', ['doc/optimization-evolution-report-20260926.md §4–5'])
#4
s=slide('02 / v1 的收益边界','重复结构不同，同一缓存策略的收益就不同','历史单轮结果：Salt 的节点复用高，Stack 的节点复用低。')
beforeafter(s,.7,2.17,5.6,'Salt：Vanilla → v1',14132.8,2832.6,('Vanilla','v1'))
beforeafter(s,7.0,2.17,5.6,'Stack：Vanilla → v1',7831.8,6428.7,('Vanilla','v1'))
text(s,.7,5.11,5.6,.85,'节点重复因子 2.114\nraw 行覆盖率 98.70%',19,c=MUTED)
text(s,7.0,5.11,5.6,.85,'节点重复因子 1.011\nraw 行覆盖率 4.86%',19,c=MUTED)
takeaway(s,'重复率决定“能省多少”；缓存查询和复制成本决定“省下来是否划算”。')
note(s,'Salt约4.99×，Stack约1.22×。历史阶段/worker计时会重叠，不求和为wall。此处仅解释v1在不同数据结构上的收益差异。',['doc/gpu-feature-optimization-experiment.md'])
#5
s=slide('03 / 又发现了什么问题','命中缓存以后，访问成本仍可能成为瓶颈','增加缓存和 worker，并不必然让训练更快。')
items=[('高命中仍然慢','HM / Salt','整块 RPC、共享锁、高频查表'),('工作集超出内存','Arxiv / Stack / Trial','缓存填满后反复查询、跨轮重算'),('并发资源持续增长','Stack / Amazon','mmap 数量、WAL、刷盘等待'),('准备工作压倒训练','Event / F1','宽表冷统计；小任务启动成本')]
for i,(a,b,c) in enumerate(items):
 x=.7+(i%2)*6.25;y=2.08+(i//2)*1.88
 text(s,x,y,5.7,.4,a,23,True);text(s,x,y+.53,5.7,.4,b,17,True,TEAL);text(s,x,y+1.01,5.7,.58,c,18,c=MUTED)
takeaway(s,'v2 需要同时解决：供给并行度、命中成本、缓存容量和长期运行资源。')
note(s,'本页是诊断归纳，不宣称每个问题只有单一原因。Ratebeer另有共享存储发布竞争，属于正确性问题，在后面独立说明。',['doc/v2-full-matrix-20260925.md','doc/v2-hm-regression-20260925.md'])
#6
s=slide('04 / V2 架构','采样、特征组装、GPU 训练分进程执行','v1 单进程优化后，assemble 供给仍受限；v2 将三个阶段拆分并重叠执行。')
node(s,.65,2.75,2.05,1.25,'采样进程','seed + 常驻拓扑')
node(s,3.08,2.9,1.55,.98,'计划队列','有界窗口')
for i in range(3):node(s,5.03,2.02+i*1.2,2.65,.98,f'Assembly {i+1 if i<2 else "N"}','SQL → 文本 → Frame')
node(s,8.10,2.9,1.72,.98,'就绪队列','准备好张量')
node(s,10.22,2.75,2.45,1.25,'GPU 主进程','前向 / 反向 / 更新')
line(s,2.72,3.35,3.04,3.35,TEAL,True)
for yy in [2.5,3.7,4.9]:line(s,4.66,3.35,5.0,yy,TEAL,True);line(s,7.72,yy,8.05,3.35,TEAL,True)
line(s,9.85,3.35,10.18,3.35,TEAL,True)
box(s,3.1,5.51,6.73,.52,LIGHT);text(s,3.28,5.60,6.35,.31,'本地 / 共享编码缓存  +  共享文本 / 磁盘向量缓存',16,True)
takeaway(s,'队列与在途窗口受限；ready 顺序减轻队头等待，但会改变 batch 更新顺序。')
note(s,'epoch gate限制当前轮训练，下一轮可预取；特征窗口不能跨epoch混合。活跃worker数可根据供给状态调整，但预先spawn的进程暂停不等于归还内存。图拓扑仍常驻。', ['relconnector_v2/relconnector_v2/runtime/process_pipeline.py','relconnector_v2/relconnector_v2/process_api.py'])
#7
s=slide('04 / 从数据形态选择策略','按可观测信号决策，而不是按数据集名称分支','以下区分“自动策略”与“实验选定配置”。')
table(s,['信号','策略','代表数据集'],[
['高文本命中 + 查询代价高','启用本地槽位索引，绕过共享查询','Salt'],
['大库 + 低内存命中 + 编码昂贵','启用持久化 embedding；复用已有缓存','Arxiv / Stack / Trial / Amazon'],
['索引低复用 / 收益变小','退出本地索引，冷却后再评估','Arxiv 等低复用阶段'],
['数值列可装入受限缓冲','按行数与预算分组，精确内存统计','Event；小表也适用'],
['写入 / 读取阶段变化','调整 WAL 回收预算','Amazon 暴露的问题'],
['编码缓存 RPC 昂贵','只返回请求行；可选择本地作用域','HM：本地 4 GiB 是选定配置'],
],[3.8,5.1,3.2],rowh=.60,fontsize=16)
takeaway(s,'F1 / Avito 不必强开磁盘缓存；Ratebeer 也不能仅因“大”就自动解决编码瓶颈。')
note(s,'HM本地4GiB是验证矩阵显式配置，不称为已实现自动选择。磁盘和本地索引是独立策略，可能分别开关。Ratebeer磁盘策略未激活，其持续编码仍是待优化点。',['relconnector_v2/relconnector_v2/features/adaptive_text_index.py','relconnector_v2/relconnector_v2/features/disk_text.py','doc/v2-hm-regression-20260925.md'])
#8
s=slide('04 / 算法 A：高复用文本','重复命中时，只查本地槽位索引','向量仍共享；本地仅保存“文本 → 已发布槽位”，避免复制整份向量。')
node(s,.7,2.23,3.0,1.55,'观察窗口','64 次调用\n≥1,024 次请求')
node(s,5.0,2.23,3.0,1.55,'自动激活','共享命中 ≥80%\n查询累计 ≥5 ms')
node(s,9.32,2.23,3.0,1.55,'本地命中','直接批量 gather\n跳过共享哈希与锁')
line(s,3.8,2.87,4.87,2.87,TEAL,True);line(s,8.12,2.87,9.19,2.87,TEAL,True)
text(s,.76,4.13,5.60,1.5,'自动退出\n连续两个窗口本地命中 <20%：\n释放索引，并进入冷却期。',20)
metric(s,7.12,4.11,5.25,'18,260 → 5,208 s','Salt 两轮完整耗时；模型输入不变\n共享查询由约 30,532 降至 802 worker 秒')
takeaway(s,'高命中率任务的重点是降低命中成本，而非继续增加编码能力。')
note(s,'每worker索引元数据预算32MiB，LRU淘汰；激活后两个窗口预热，反复失效延长冷却，最多32个窗口。Salt端到端3.51×包含初始化状态差异；查询计时是并行累计。',['doc/v2-salt-regression-20260926.md'])
#9
s=slide('04 / 算法 B：文本工作集较大','把跨轮可复用的 embedding 留在磁盘','本地索引 → 共享文本缓存 → 磁盘缓存 → 真正缺失才编码。')
for i,label in enumerate(['本地槽位','共享向量','磁盘向量','GloVe 编码']):
 node(s,.75+i*3.15,2.18,2.65,1.0,label,'命中即返回' if i<3 else '发布结果并复用')
 if i<3:line(s,3.48+i*3.15,2.68,3.82+i*3.15,2.68,TEAL,True)
text(s,.8,3.75,6.0,1.95,'新缓存的启用条件\n数据库 ≥256 MiB；模型输入 ≥4,096；\n模型耗时 ≥1 s；内存命中率 ≤10%。\n已有兼容磁盘缓存可直接复用。',19)
text(s,7.15,3.75,5.3,1.95,'观察到的收益\nArxiv：第二轮编码显著减少。\nTrial：热缓存复测两轮编码均为 0。\nF1：小数据集不激活磁盘路径。',19)
takeaway(s,'冷构建要付编码与写入成本；热缓存成绩必须与冷启动成绩分开。')
note(s,'Arxiv矩阵模型输入184724/1316。Trial371.80秒为热初始化/文本缓存，不能全部归因于代码优化。Ratebeer内存累计命中约88.67%，未触发低命中门槛，第二轮仍有大量编码。',['doc/v2-full-matrix-20260925-final.md','doc/v2-trial-rerun-20260927.md'])
#10
s=slide('04 / 编码特征缓存：HM 的配置选择','只传需要的行，避免命中后搬运整块特征','先优化通用共享路径，再用对照实验选择本地或共享作用域。')
node(s,.8,2.2,5.5,1.45,'原共享路径','少量请求 → 整块 TensorFrame RPC → 再切片')
node(s,7.0,2.2,5.5,1.45,'优化后','返回请求行；HM 使用 worker 本地缓存')
line(s,6.38,2.92,6.91,2.92,TEAL,True)
beforeafter(s,.83,4.02,5.45,'HM：两轮流水线',4603.0,1756.3)
text(s,7.0,4.02,5.5,1.85,'等待 ready：3,348.9 → 481.9 秒\n文本编码、SQL 计时基本不变。\n\n使用本地总计 4 GiB、4 worker；\n这是验证过的配置，不是自动决策。',18)
note(s,'旧shared2GiB vs local总4GiB并非等容量单因素对照；报告另有同预算短测。pipeline快2.62×，端到端快2.87×包含热初始化收益。',['doc/v2-hm-regression-20260925.md'])
#11
s=slide('04 / 算法 C：控制并发资源成本','缓存能命中，还要能长期稳定运行','缩短临界区、限制映射数量，并随读写阶段调整日志回收。')
text(s,.75,2.15,5.7,.42,'Stack｜查询与映射',23,True)
text(s,.75,2.85,5.7,2.63,'• 哈希、gather 移到锁外，批量发布。\n• mmap 最多保留 1,024 个，LRU 回收。\n• 共享查询短测：296.0 → 53.5 秒。\n• 完整两轮：3,665.66 秒。',20)
text(s,7.0,2.15,5.6,.42,'Amazon｜WAL 与写入',23,True)
text(s,7.0,2.85,5.6,2.63,'• 每 writer 追加 8,192 行固定分片。\n• 预算参考索引大小和剩余磁盘。\n• 读多写少时降低 WAL 回收阈值。\n• 避免固定小阈值造成频繁刷盘。',20)
takeaway(s,'WAL 阈值是回收触发条件，不是硬上限；Amazon 完整成绩未包含后加读阶段策略。')
note(s,'写阶段预算参考4倍索引、通常64MiB至4GiB并受空闲空间1/16约束。连续两窗口高命中、低WAL增长进入读阶段，上限512MiB；命中下降或增长回升退出。Amazon约60GB WAL已收回；不能把稳定性修复与性能贡献混为单一倍率。',['doc/v2-stack-regression-20260926.md','doc/v2-amazon-regression-20260926.md'])
#12
s=slide('04 / 算法 D：冷初始化数值统计','根据行数与内存预算，自动合并数值列扫描','不按 Event 特判，不抽样；精确统计结果保留。')
text(s,.78,2.05,6.05,2.82,'1  估算单列缓冲：行数 × 8 字节\n2  分组列数 = min(32, 128 MiB / 单列大小)\n3  合并读取，批次最多 8,192 行\n4  内存 partition 选取精确分位点\n5  单列超预算 → 回退流式 SQL 路径',19)
beforeafter(s,7.15,2.12,5.0,'Event：完整冷初始化',2164.0,606.01)
text(s,.80,5.15,11.7,.78,'Schema：1,895.62 → 335.66 秒；图拓扑：267.95 → 269.92 秒。\n329 项统计核对：分位数完全一致；均值/标准差最大绝对差约 8×10⁻¹³。',18)
takeaway(s,'总初始化减少 72.0%（3.57×）；剩余约 44.5% 时间在图拓扑构建。')
note(s,'空应用缓存对照，不清操作系统文件缓存。预算只针对数值数组，非进程RSS上限。schema缓存版本更新，旧schema需重算；图缓存可复用。新旧完整冷测分别2164.0016/606.0092秒。',['doc/v2-initialization-optimization-20260927.md','benchmarks/v2-event-init-optimized-20260927/result.json'])
#13
s=slide('04 / 正确性是优化的前提','Ratebeer：先修复发布竞争，再评价吞吐','并发修复解决数据损坏，不应直接包装成性能加速算法。')
node(s,.75,2.20,5.55,1.55,'旧发布顺序','索引可见 → RPC 搬迁 storage\n并发读可能访问旧地址')
node(s,7.0,2.20,5.55,1.55,'新发布顺序','先完成共享 storage → 再发布索引\n后续序列化不搬迁已发布对象')
line(s,6.39,2.96,6.90,2.96,TEAL,True)
text(s,.8,4.2,11.6,1.5,'验证：独立协调复现 + 回归测试 + 两轮各 10,000 batch 完成。\n保留失败输入、精确 batch key 和类别校验；不截断坏值、不跳过样本。\n仍有大量文本编码、SQL 和组装开销，磁盘策略未启用。',20)
takeaway(s,'首个 CUDA 异常未保存输入，不能确认与后续已复现竞争完全同源。')
note(s,'Ratebeer完整结果15444.93秒，包含CPU类别检查开销；20k批为截断工作量。最终模型输入约20245311/20503129，ready等待11389.19秒。',['doc/v2-ratebeer-regression-20260926.md'])
#14
s=slide('05 / 实验总结','十项已有有效成绩，但并非统一最终版本重测','最新 v2 结果：Event / F1 用初始化优化后验证，Trial 用热缓存复测，其余沿用矩阵。')
cachelabels={'amazon':'部分热文本；诊断缺失','arxiv':'已有磁盘 embedding','avito':'磁盘文本关闭','event':'热初始化；非冷启动','f1':'空初始化缓存验证','hm':'本地编码缓存 4 GiB','ratebeer':'每轮截断 10,000 批','salt':'adaptive 索引复测','stack':'已有磁盘 embedding','trial':'初始化 + 文本热缓存'}
rows=[]
for k in ORDER:
 r=DATA[k];rows.append([NAMES[k],f"{r['v2_batches']:,}",f"{r['v2_total_s']:,.2f}",f"{r['v2_s_per_batch']:.3f}",cachelabels[k]])
table(s,['数据集','完成批次','端到端 / s','s / batch','解释条件'],rows,[1.4,1.8,2.25,1.9,4.85],y=2.0,rowh=.365,fontsize=15)
takeaway(s,'初始化优化：59 项测试通过；Event / F1 两轮训练通过。跨版本训练等价仍需独立验收。')
note(s,'所有结果来自原始JSONL并核对两轮步数和有限loss。历史数据来自报告，旧JSONL不在工作区。F1最新28.11秒而非挑旧26.57最低值；不要声称每次新改动都让每个任务更快。',['doc/v2-dataset-performance-review-20260927.md','benchmarks/v2-performance-review-20260927/comparison.json'])
#15
s=slide('06 / 耗时对比 · 1 / 2','大规模任务：对比 Vanilla、v1 与 v2','统一为端到端秒 / batch；历史 1 epoch，v2 2 epoch，启动成本摊销不同。')
s.shapes.add_picture(str(ASSETS/'performance_a.png'),Inches(.57),Inches(1.91),width=Inches(12.16),height=Inches(4.50))
text(s,.74,6.52,11.9,.48,'斜线 Vanilla 条为超时下界（>）；Ratebeer v1 为超时吞吐估算（≈）。',16,c=MUTED)
note(s,'图中无Batch版本。Vanilla超时值=14400/计划完整epoch批数，仅是完整epoch平均秒/batch下界，非实测完成值。Ratebeer v1=14400/15686，v2每轮限10000批。Arxiv/Amazon已有文本缓存；版本和冷热不配对。',['doc/gpu-feature-optimization-experiment.md','benchmarks/v2-performance-review-20260927/comparison.json'])
#16
s=slide('06 / 耗时对比 · 2 / 2','缓存与初始化影响，必须和吞吐一起解释','秒 / batch 越低越好；跨实验比较展示观测结果，不单独证明算法因果收益。')
s.shapes.add_picture(str(ASSETS/'performance_b.png'),Inches(.57),Inches(1.91),width=Inches(12.16),height=Inches(4.50))
text(s,.74,6.52,11.9,.48,'† Event 是热初始化训练，Trial 复用文本缓存；Event 冷初始化独立提升为 3.57×。',16,c=MUTED)
note(s,'Event38.77秒/76批，Trial371.80秒/918批。Event的49.83×跨实验单位batch比值不是冷初始化加速，也不是训练算子加速。本图刻意不以该倍率做宣传标题。Salt/Stack也有各自缓存和代码修订条件。',['doc/v2-dataset-performance-review-20260927.md','doc/v2-initialization-optimization-20260927.md'])
#17
s=slide('06 / 哪些证据能支持优化机制','用局部对照解释收益，而非只看版本排名','不同实验各有边界，不能把局部倍数直接相乘。')
table(s,['瓶颈','对照证据','能说明什么'],[
['HM：跨进程传输','流水线 4,603 → 1,756 秒','减少整块 RPC / 本地作用域有效'],
['Salt：共享文本查询','完整两轮 18,260 → 5,208 秒','模型输入不变，命中路径成本下降'],
['Stack：共享查询临界区','同规模短测 296.0 → 53.5 秒','批量化与锁外处理改善缓存访问'],
['Event：精确数值统计','冷初始化 2,164 → 606 秒','空缓存下确有算法收益，结果一致'],
['Trial：已有文本缓存复用','两轮 656.95 → 371.80 秒','收益包含热缓存，不能全归因于改代码'],
],[3.0,4.45,4.65],rowh=.71,fontsize=17)
takeaway(s,'观察：选定 v2 成绩均低于历史 v1；因果结论优先依赖明确的阶段对照。')
note(s,'HM作用域和容量不同，Salt初始化可能不同，Stack仅短测，EventOS页缓存不清。准确区分测量边界。所有worker累计计时不能直接相加为wall。',['doc/v2-hm-regression-20260925.md','doc/v2-salt-regression-20260926.md','doc/v2-stack-regression-20260926.md','doc/v2-trial-rerun-20260927.md'])
#18
s=slide('07 / RSS 对比 · 包含 Batch','全量物化的内存代价明显；v2 总内存尚待补测','实色：历史报告峰值 RSS；斜线：v2 主进程 RSS，不包含采样 / 组装 / manager。')
s.shapes.add_picture(str(ASSETS/'rss.png'),Inches(.57),Inches(1.93),width=Inches(12.16),height=Inches(4.48))
text(s,.74,6.52,11.9,.48,'不能由 v2 主进程约 2–3.3 GiB 推断整机占用，也不据此计算相对 Batch 的节省比例。',16,True,ORANGE)
note(s,'历史RSS（GiB）来自gpu-feature-optimization-experiment.md十库汇总；超时任务RSS只是截至超时的峰值。v2取与耗时图相同JSONL的peak_rss_mb/1024，其源码是RUSAGE_SELF。非同口径四版本总内存比较；使用斜线并明确标注，不进行内存排名。完整数据见rss-data.csv。',['doc/gpu-feature-optimization-experiment.md','relconnector_v2/relconnector_v2/benchmark.py','rss-data.csv'])
#19
s=slide('07 / 内存结论与下一步','减少常驻特征，还需要覆盖整个进程树','缓存预算、队列上限与主进程 RSS，都不等于系统总内存。')
for i,(a,b) in enumerate([('训练主进程','模型 / 优化器 / 当前 batch'),('采样 + manager','拓扑 / 索引 / 共享缓存'),('Assembly × N','GloVe / SQL / 临时 Frame')]):node(s,.7+i*4.16,2.2,3.68,1.55,a,b)
text(s,.8,4.2,11.7,1.72,'测量补齐：进程树 PSS / USS、cgroup 峰值、显存与磁盘资源。\n性能优先级：Amazon / Ratebeer 的特征供给；Event 的图拓扑构建。\n最终总评：固定代码和资源预算，分别做冷缓存与热缓存十库复测。',20)
takeaway(s,'主线不变：少做重复工作 → 提高供给 → 让命中更便宜 → 控制资源与冷启动。')
note(s,'不直接相加进程RSS，因为共享页会重复计算。v2默认ready队列16GiB为上限而非实际占用，也非全系统内存承诺。通用启发式已实现，但HM本地配置不是自动学习，所有十库尚未用新统计代码重测。',['doc/optimization-evolution-report-20260926.md §9','doc/v2-dataset-performance-review-20260927.md'])

filename=OUT/'RelConnector-瓶颈驱动优化复盘.pptx'
R.core_properties.title='RelConnector：瓶颈驱动的优化演进'
R.core_properties.subject='Vanilla → v1 → v2；架构、启发式、性能与RSS'
R.core_properties.author='RelConnector'
R.save(filename)
(OUT/'speaker-notes.md').write_text('# 逐页讲稿与证据\n\n'+'\n\n'.join(f"## {n['page']:02} {n['title']}\n\n{n['notes']}" for n in NOTES))
(OUT/'slide-manifest.json').write_text(json.dumps(NOTES,ensure_ascii=False,indent=2)+'\n')
# The measurements used by every performance chart are copied alongside the deck.
(OUT/'performance-data.json').write_text(json.dumps(ROWS,ensure_ascii=False,indent=2)+'\n')
print(filename)
print('slides',len(R.slides))
