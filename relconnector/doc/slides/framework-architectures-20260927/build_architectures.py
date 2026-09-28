"""Four source-verified, native PowerPoint / SVG architecture figures."""
from pathlib import Path
from copy import deepcopy
from html import escape
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from PIL import ImageFont

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]
W,H=1200,600
S=12.30/W
OX,OY=.515,1.0
INK='263746';CPU='365F85';DATA='EEF4FA';CACHE='EAF5EF';GREEN='39715A';GPU='FFF2E1';GOLD='9A692D';GRAY='81909C';PALE='F8FAFC';LINE='B5C0C9';WHITE='FFFFFF'
FONTPATH='/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'
BOLDPATH='/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'
R=Presentation();R.slide_width=Inches(13.333);R.slide_height=Inches(7.5)
R.core_properties.title='RelConnector — Four Training Architectures'
R.core_properties.subject='Batch / Vanilla / v1 / v2; editable vector diagrams'
R.core_properties.author='RelConnector'
MANIFEST=[]

def rgb(c):return RGBColor.from_string(c)
def flat(sh):
 sp=sh._element.spPr
 sp.append(OxmlElement('a:effectLst'))
 style=sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
 if style is not None:
  ref=style.find('{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef')
  if ref is not None:ref.set('idx','0')
 return sh

def ppt_text(s,x,y,w,h,value,pt=18,bold=False,c=INK,font='Arial',align=PP_ALIGN.LEFT):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=False
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(value.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=font;p.font.size=Pt(pt);p.font.bold=bold;p.font.color.rgb=rgb(c);p.alignment=align;p.space_before=Pt(0);p.space_after=Pt(0);p.line_spacing=1.12
  for run in p.runs:
   ea=OxmlElement('a:ea');ea.set('typeface',font);run._r.get_or_add_rPr().append(ea)
 return sh

class Figure:
 def __init__(self,key,title,subtitle,notes):
  self.key=key;self.svg=[]
  self.s=R.slides.add_slide(R.slide_layouts[6]);self.s.background.fill.solid();self.s.background.fill.fore_color.rgb=rgb(WHITE)
  ppt_text(self.s,.52,.21,12.1,.42,title,24,True,font='Noto Sans CJK SC')
  ppt_text(self.s,.54,.72,12.05,.22,subtitle,10.5,c=GRAY,font='Noto Sans CJK SC')
  self.s.notes_slide.notes_text_frame.text=notes
  MANIFEST.append({'figure':key,'title':title,'caption':subtitle,'notes':notes})
 def rect(self,x,y,w,h,fill=WHITE,stroke=LINE,dash=False,radius=0,lw=1.3):
  sh=self.s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(OX+x*S),Inches(OY+y*S),Inches(w*S),Inches(h*S));flat(sh)
  if radius:sh.adjustments[0]=min(radius/min(w,h),.12)
  sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
  sh.line.color.rgb=rgb(stroke);sh.line.width=Pt(lw*S*72)
  if dash:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
  self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="#{fill}" stroke="#{stroke}" stroke-width="{lw}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
  return sh
 def text(self,x,y,w,lines,size=19,bold=False,c=INK,align='center'):
  lines=lines.split('\n') if isinstance(lines,str) else lines
  face=BOLDPATH if bold else FONTPATH
  # Fit explicit lines only; prevent auto-wrapping and accidental overlap.
  while max(ImageFont.truetype(face,round(size*4)).getlength(t)/4 for t in lines)>w and size>12:size-=.5
  ppt_text(self.s,OX+x*S,OY+y*S,w*S,(len(lines)*size*1.18+3)*S,'\n'.join(lines),size*S*72,bold,c,align=PP_ALIGN.CENTER if align=='center' else PP_ALIGN.LEFT)
  anchor='middle' if align=='center' else 'start';xx=x+w/2 if align=='center' else x
  for i,t in enumerate(lines):
   self.svg.append(f'<text x="{xx}" y="{y+size*.92+i*size*1.12}" font-family="Arial, Liberation Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="#{c}" text-anchor="{anchor}">{escape(t)}</text>')
 def edge(self,pts,c=INK,dash=False,both=False,arrow=True,width=1.7):
  for i,(p,q) in enumerate(zip(pts,pts[1:])):
   sh=self.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(OX+p[0]*S),Inches(OY+p[1]*S),Inches(OX+q[0]*S),Inches(OY+q[1]*S));flat(sh);sh.line.color.rgb=rgb(c);sh.line.width=Pt(width*S*72)
   if dash:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
   ln=sh._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
   if arrow and i==len(pts)-2:
    e=OxmlElement('a:tailEnd');e.set('type','triangle');e.set('w','sm');e.set('len','sm');ln.append(e)
   if both and i==0:
    e=OxmlElement('a:headEnd');e.set('type','triangle');e.set('w','sm');e.set('len','sm');ln.append(e)
  mark='cache' if c==GREEN else 'control' if c==GOLD else 'meta' if c==GRAY else 'data'
  p=' '.join(f'{x},{y}' for x,y in pts)
  self.svg.append(f'<polyline points="{p}" fill="none" stroke="#{c}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="6 4"' if dash else '')+(f' marker-end="url(#{mark})"' if arrow else '')+(f' marker-start="url(#{mark}start)"' if both else '')+'/>')
 def boundary(self,x,y,w,h,label,c=GRAY,fill=WHITE):
  self.rect(x,y,w,h,fill,c,True,7,1.1);self.text(x+12,y+9,w-24,label,17,True,c,'left')
 def block(self,x,y,w,h,title,sub='',fill=DATA,stroke=CPU,size=19):
  self.rect(x,y,w,h,fill,stroke,False,6,1.3)
  lines=title.split('\n');top=y+12
  self.text(x+9,top,w-18,lines,size,True)
  if sub:self.text(x+9,y+h-12-len(sub.split('\n'))*17*1.12,w-18,sub,17,c=GRAY)
 def store(self,x,y,w,h,title,sub='',persistent=False,fill=CACHE,stroke=GREEN):
  if persistent:
   sh=self.s.shapes.add_shape(MSO_SHAPE.CAN,Inches(OX+x*S),Inches(OY+y*S),Inches(w*S),Inches(h*S));flat(sh);sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill);sh.line.color.rgb=rgb(stroke);sh.line.width=Pt(1.2*S*72)
   # Match the native cylinder with explicit SVG geometry.
   ry=8
   self.svg.append(f'<path d="M{x},{y+ry} A{w/2},{ry} 0 0 1 {x+w},{y+ry} L{x+w},{y+h-ry} A{w/2},{ry} 0 0 1 {x},{y+h-ry} Z" fill="#{fill}" stroke="#{stroke}" stroke-width="1.2"/>')
   self.svg.append(f'<ellipse cx="{x+w/2}" cy="{y+ry}" rx="{w/2}" ry="{ry}" fill="#{fill}" stroke="#{stroke}" stroke-width="1.2"/>')
   self.text(x+8,y+(15 if h<60 else 19),w-16,title,16 if h<60 else 18,True)
   if sub:self.text(x+8,y+h-(18 if h<60 else 23),w-16,sub,12 if h<60 else 15,c=GRAY)
  else:
   if h<60:
    self.rect(x,y,w,h,fill,stroke,False,6,1.3)
    self.text(x+8,y+6,w-16,title,16,True)
    if sub:self.text(x+8,y+h-19,w-16,sub,12,c=GRAY)
   else:self.block(x,y,w,h,title,sub,fill,stroke,18)
 def queue(self,x,y,h,label):
  w=29
  self.rect(x,y,w,h,WHITE,CPU,False,2)
  for yy in [y+12,y+24,y+36]:self.edge([(x+6,yy),(x+w-6,yy)],c=CPU,arrow=False,width=1.2)
  self.text(x-18,y+h+9,w+36,label,14,c=CPU)
 def dot(self,x,y,c=INK):
  sh=self.s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(OX+(x-2.7)*S),Inches(OY+(y-2.7)*S),Inches(5.4*S),Inches(5.4*S));flat(sh);sh.fill.solid();sh.fill.fore_color.rgb=rgb(c);sh.line.fill.background()
  self.svg.append(f'<circle cx="{x}" cy="{y}" r="2.7" fill="#{c}"/>')
 def trainer(self,label='Trainer / GPU'):
  self.boundary(970,282,210,265,label,GOLD,'FFFCF6')
  self.block(986,329,168,52,'Feature enc. + GNN',fill=GPU,stroke=GOLD,size=17.5)
  self.block(986,405,168,45,'Loss + backward',fill=GPU,stroke=GOLD,size=18)
  self.block(986,477,168,45,'Optimizer step',fill=GPU,stroke=GOLD,size=18)
  self.edge([(1070,381),(1070,405)],c=GOLD)
  self.edge([(1070,450),(1070,477)],c=GOLD)
  self.edge([(1154,499),(1166,499),(1166,355),(1154,355)],c=GOLD,dash=True)
  self.text(1104,304,56,'θ',18,c=GOLD)
 def common_inputs(self):
  self.store(18,73,145,77,'Relational DB','tables + catalog',True,DATA,CPU)
  self.store(18,335,145,78,'Task seeds','IDs / times / labels',True,DATA,CPU)
 def legend(self,desc):
  self.edge([(200,584),(238,584)],arrow=True);self.text(248,574,120,'data flow',14,c=GRAY,align='left')
  self.edge([(379,584),(417,584)],c=GREEN,both=True);self.text(427,574,120,'cache access',14,c=GRAY,align='left')
  self.edge([(570,584),(608,584)],c=GRAY,dash=True);self.text(618,574,185,'metadata / control',14,c=GRAY,align='left')
  self.text(18,575,150,desc,13,c=GRAY,align='left')
 def finish(self,caption):
  defs=[]
  for name,c in [('data',INK),('cache',GREEN),('control',GOLD),('meta',GRAY)]:
   defs.append(f'<marker id="{name}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#{c}"/></marker>')
   defs.append(f'<marker id="{name}start" markerWidth="8" markerHeight="8" refX="1" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M8,0 L0,4 L8,8 Z" fill="#{c}"/></marker>')
  doc=f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600" role="img" aria-label="{self.key} training architecture"><title>{escape(caption)}</title><defs>'+''.join(defs)+'</defs><rect width="1200" height="600" fill="white"/>'+''.join(self.svg)+'</svg>'
  (OUT/f'{self.key}.svg').write_text(doc)
  ppt_text(self.s,.54,7.23,12.15,.16,caption,8.3,c=GRAY)

# A — full materialization; the training path has no feature SQL queries.
f=Figure('a-batch','(a) Batch｜全量物化训练架构','训练前生成全部特征；训练时从常驻 TensorFrame 切片。虚线框表示执行范围。',
'''Source: benchmark/baseline.py; baseline/batch_baseline/feature_store.py. Full table reads and EagerFeatureStore.materialize precede training. Graph topology is resident. The async baseline reuses sampler/feature threads, but feature fetch is an in-memory TensorFrame slice. The node/edge/time/label information in SamplePlan is preserved during assembly. Feature enc. in the GPU block is the trainable model encoder, distinct from frozen GloVe/statistical preprocessing. Task-seed queue is omitted here for clarity.\n中文：全量物化不等于全批量梯度下降；本架构仍采用邻居采样和mini-batch训练。''')
f.boundary(190,14,760,538,'One host process · preparation + threaded mini-batch pipeline')
f.common_inputs();f.trainer()
f.block(224,74,183,67,'Schema + topology','statistics / PK–FK')
f.block(482,74,180,67,'Encode all rows','frozen text + features')
f.store(713,74,209,67,'Full TensorFrames','resident across epochs')
f.store(224,187,183,67,'Graph / time index','resident in CPU RAM',fill=DATA,stroke=CPU)
f.boundary(205,299,180,145,'Sampler thread',fill=PALE)
f.block(216,330,158,84,'Temporal neighbor\nsampling',size=18)
f.boundary(455,299,430,145,'Feature thread · in-memory gather',fill=PALE)
f.block(470,330,170,84,'TensorFrame slice','sampled node IDs')
f.block(700,330,170,84,'Batch assembly','edges + times + labels')
f.queue(402,341,63,'plans');f.queue(910,341,63,'ready')
f.edge([(163,110),(224,110)])
f.edge([(90,73),(90,45),(572,45),(572,74)]);f.text(420,51,200,'all feature rows',14,c=GRAY)
f.edge([(407,108),(482,108)],c=GRAY,dash=True)
f.edge([(662,108),(713,108)])
f.edge([(316,141),(316,187)])
f.edge([(316,254),(316,280),(392,280),(392,345),(374,345)])
f.edge([(817,141),(817,271),(555,271),(555,299)],c=GREEN)
f.text(627,246,160,'row selection',14,c=GREEN)
f.edge([(163,372),(216,372)])
f.edge([(374,372),(402,372)]);f.edge([(431,372),(470,372)])
f.edge([(640,372),(700,372)]);f.edge([(870,372),(910,372)]);f.edge([(939,372),(955,372),(955,355),(986,355)])
f.text(462,469,450,['Hot path: slice precomputed features; no feature SQL.','Memory scales with the full encoded feature store.'],17,c=GRAY,align='left')
f.legend('Eager feature store')
f.finish('Batch: eager materialization followed by sampled mini-batch training. Static features are resident; model parameters remain trainable.')

# B — Vanilla: existing threads and raw cache, no later text/encoded caches.
f=Figure('b-vanilla','(b) Vanilla｜按需读取的在线训练架构','单进程内异步采样、特征线程与训练；已有 raw cache，但缺少后续文本与编码特征复用。',
'''Source: baseline/vanilla_baseline/api.py and runtime/async_pipeline.py. The default async path already has seed/sampler/feature threads and bounded queues. It is not a fully serial or cache-free implementation. SqlFeatureFetcher owns a bounded raw block cache. GloveTextEmbedder is constructed without the later text-cache options; no encoded-feature cache is configured. The repeated SQL-tables cylinder denotes the same database, not another copy. The initial schema statistics configure the static feature encoder. Recommendation branch expansion and error channels are omitted for clarity.''')
f.boundary(190,14,760,538,'One host process · asynchronous threads')
f.common_inputs();f.trainer()
f.block(224,74,183,67,'Schema + topology','task-aware preparation')
f.store(224,187,183,67,'Graph / time index','resident in CPU RAM',fill=DATA,stroke=CPU)
f.store(467,187,175,67,'Raw block cache','bounded DataFrames')
f.boundary(205,299,180,145,'Sampler thread',fill=PALE)
f.block(216,330,158,84,'Temporal neighbor\nsampling',size=18)
f.boundary(455,299,430,145,'Single feature thread',fill=PALE)
f.block(470,330,170,84,'SQL fetch + decode','requested node rows')
f.block(700,330,170,84,'Encode + assemble','text → TensorFrame')
f.queue(402,341,63,'plans');f.queue(910,341,63,'ready')
f.store(467,478,175,67,'SQL tables','same relational DB',True,DATA,CPU)
f.edge([(163,110),(224,110)])
f.edge([(316,141),(316,187)])
f.edge([(316,254),(316,280),(392,280),(392,345),(374,345)])
f.edge([(316,141),(316,161),(884,161),(884,279),(785,279),(785,330)],c=GRAY,dash=True);f.text(650,141,190,'types + statistics',14,c=GRAY)
f.edge([(554,254),(554,299)],c=GREEN,both=True)
f.edge([(554,478),(554,414)]);f.text(568,448,90,'on demand',13,c=GRAY,align='left')
f.edge([(163,372),(216,372)]);f.edge([(374,372),(402,372)]);f.edge([(431,372),(470,372)])
f.edge([(640,372),(700,372)]);f.edge([(870,372),(910,372)]);f.edge([(939,372),(955,372),(955,355),(986,355)])
f.text(697,477,220,['Repeated text conversion','and feature assembly','remain on the hot path.'],17,c=GRAY,align='left')
f.legend('On-demand features')
f.finish('Vanilla: bounded online feature access with a raw cache and asynchronous threads; static text / encoded-feature reuse is not yet added.')

# C — v1: windows, in-process layered caches, split fetch / encoding threads.
f=Figure('c-v1','(c) v1｜单进程优化：窗口复用与分层缓存','在单进程线程流水线中合并请求，并复用 raw、text 与 encoded 特征；训练按序消费。',
'''Source: relconnector/api.py, runtime/async_pipeline.py, features/sql.py and features/assembler.py. This depicts the current legacy/v1 architecture, including later direct-text/thread refinements; it is not a claim that every component was present in the historical performance snapshot. Feature windows do not cross epochs. Requested node rows are deduplicated and restored using inverse mappings. Encoded-cache hits reuse requested encoded rows; missing rows follow raw/text/encoding paths. Encoder workers are threads sharing process-local caches, not v2 assembly processes. Persisted schema/topology artifacts are initialization caches, not a per-text disk cache.''')
f.boundary(190,14,760,538,'One host process · fetch / encode overlap and ordered consumption')
f.common_inputs();f.trainer()
f.block(224,74,183,67,'Schema + topology','build or load artifacts')
f.store(472,74,360,67,'Initialization artifacts','persisted schema + graph',True)
f.store(224,187,183,67,'Graph / time index','resident in CPU RAM',fill=DATA,stroke=CPU)
f.store(464,187,163,67,'Raw cache','decoded blocks')
f.store(650,187,155,67,'Text cache','frozen vectors')
f.store(828,187,112,67,'Encoded cache','static TF rows')
f.boundary(205,299,180,145,'Sampler thread',fill=PALE)
f.block(216,330,158,84,'Temporal neighbor\nsampling',size=18)
f.boundary(443,291,195,160,'Fetch thread',fill=PALE)
f.block(454,326,173,106,'Window + SQL','unique IDs\nsparse / block reads',size=18)
f.boundary(703,291,182,173,'Encode threads',fill=PALE)
f.block(714,327,160,44,'Direct text + Frame',size=16.5)
f.block(714,385,160,44,'Restore batch order',size=16.5)
f.text(714,440,160,'1 ... K workers',14,c=GRAY)
f.queue(402,341,63,'plans');f.queue(654,341,63,'fetched');f.queue(910,341,63,'ordered')
f.store(464,479,173,67,'SQL tables','cache misses only',True,DATA,CPU)
f.edge([(163,110),(224,110)]);f.edge([(407,108),(472,108)],c=GREEN,both=True)
f.edge([(316,141),(316,187)]);f.edge([(316,254),(316,280),(392,280),(392,345),(374,345)])
f.edge([(545,254),(545,291)],c=GREEN,both=True)
f.edge([(727,254),(727,279),(749,279),(749,291)],c=GREEN,both=True)
f.edge([(884,254),(884,273),(861,273),(861,291)],c=GREEN,both=True)
f.edge([(550,479),(550,432)])
f.edge([(163,372),(216,372)]);f.edge([(374,372),(402,372)]);f.edge([(431,372),(454,372)])
f.edge([(627,372),(654,372)]);f.edge([(683,372),(714,349)])
f.edge([(794,371),(794,385)]);f.edge([(874,407),(896,407),(896,372),(910,372)])
f.edge([(939,372),(955,372),(955,355),(986,355)])
f.text(693,476,237,['Adaptive window size','Byte-bounded queues','Static / frozen caches only'],16,c=GRAY,align='left')
f.edge([(365,141),(365,160),(637,160),(637,270),(680,270),(680,303),(703,303)],c=GRAY,dash=True)
f.text(421,141,198,'types + statistics',14,c=GRAY)
f.legend('Reuse before encode')
f.finish('v1: deduplicated feature windows, layered in-process caches, overlapping fetch / encoding threads, and restored per-batch inputs.')

# D — v2: separate sampling / assembly processes, shared caches and disk embedding.
f=Figure('d-v2','(d) v2｜多进程与自适应缓存训练架构','assemble 成为供给瓶颈后，v2 将采样、组装、训练分进程执行。',
'''Source: relconnector_v2/process_api.py, runtime/process_pipeline.py, features/process_worker.py, features/shared_text.py, features/shared_encoded.py, features/disk_text.py. The seed feeder is a thread in the main process; its input stream is represented at the left. Sampling is a child process. Assembly workers own SQL readers, raw caches, text encoder and local state. Shared text vectors have reservation-based coordination; the adaptive local index stores only text-to-slot metadata. Encoded-feature cache is optionally shared through a manager or local to each worker; HM's local 4 GiB is an explicit tested profile, not an automatic learned choice. Disk text persistence and initialization artifacts are separate stores. CPU topology is resident; do not claim a single shared physical graph allocation. CPU initialization includes budgeted exact numerical statistics. Training may use ready-order or strict-order with an epoch gate. Queue byte caps / in-flight windows impose backpressure. Diagram excludes error-reporting paths and checkpoint persistence for readability.''')
f.common_inputs()
f.boundary(190,14,760,135,'Parent preparation · budgeted exact statistics / graph construction')
f.block(210,62,185,67,'Schema + topology','types / statistics / PK–FK')
f.store(479,62,311,67,'Initialization artifacts','persisted schema + graph',True)
f.store(211,184,177,67,'Graph / time index','resident for sampling',fill=DATA,stroke=CPU)
f.store(479,179,181,67,'Encoded-feature cache','shared manager OR local')
f.store(681,179,232,67,'Shared text vectors','reservations + local slot index')
f.boundary(201,294,181,146,'Sampler process',fill=PALE)
f.block(214,332,155,80,'Temporal neighbor\nsampling',size=18)
f.queue(168,343,59,'seeds');f.queue(407,343,59,'plans')
f.boundary(473,281,427,214,'Assembly processes × N',CPU,PALE)
for i,(y,label) in enumerate([(318,'Assembly 1'),(376,'Assembly 2'),(434,'Assembly N')]):
 f.block(491,y,392,45,f'{label}    SQL → encode → Frame',size=18)
f.text(489,480,396,'Worker-local SQL / raw cache / encoder state',12,c=GRAY)
f.queue(919,343,59,'ready')
f.trainer('Main process / GPU')
f.store(211,510,177,53,'SQL tables','same relational DB',True,DATA,CPU)
f.store(512,517,333,45,'Persistent text embeddings','adaptive SQL index + vector shards',False)
f.rect(968,559,212,30,WHITE,GRAY,radius=6)
f.text(977,566,194,'Epoch gate / worker gate',14,True)
# Init and streams. Metadata route stays above cache stores.
f.edge([(163,110),(210,96)])
f.edge([(395,96),(479,96)],c=GREEN,both=True)
f.edge([(301,129),(301,184)])
f.edge([(301,251),(301,269),(392,269),(392,344),(369,344)])
f.edge([(163,372),(168,372)]);f.edge([(197,372),(214,372)])
f.edge([(369,372),(407,372)])
# Plan dispatch and ready gather use separated orthogonal buses.
f.edge([(436,372),(458,372)],arrow=False)
f.edge([(458,340),(458,457)],arrow=False)
for y in [340,398,456]:
 f.edge([(458,y),(491,y)])
 f.edge([(883,y),(905,y)],arrow=False)
f.edge([(905,340),(905,456)],arrow=False)
f.edge([(905,372),(919,372)])
f.dot(458,372);f.dot(905,372)
f.edge([(948,372),(957,372),(957,355),(986,355)])
f.edge([(569,246),(569,281)],c=GREEN,both=True)
f.edge([(797,246),(797,281)],c=GREEN,both=True)
f.edge([(388,539),(452,539),(452,476),(473,476)])
f.edge([(681,495),(681,517)],c=GREEN,both=True)
f.edge([(1074,559),(1074,547)],c=GRAY,dash=True)
f.edge([(351,129),(351,160),(932,160),(932,267),(883,267),(883,281)],c=GRAY,dash=True)

f.legend('Parallel + adaptive')
f.finish('v2 upgrades v1 after assembly becomes a bottleneck: sampling, assembly and training run in separate processes connected by bounded queues.')

# Save four native vector slides, SVGs, captions and integrated deck.
R.save(OUT/'RelConnector-四种训练框架架构图.pptx')
(OUT/'captions-and-sources.md').write_text('# Four training architectures\n\n'+'\n\n'.join(f"## {m['title']}\n\n{m['caption']}\n\n{m['notes']}" for m in MANIFEST))
(OUT/'manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+'\n')
main=Presentation(BASE/'doc/slides/bottleneck-review-20260927/RelConnector-瓶颈驱动优化复盘.pptx')
for src in R.slides:
 dst=main.slides.add_slide(main.slide_layouts[6])
 for sh in src.shapes:
  dst.shapes._spTree.insert_element_before(deepcopy(sh.element),'p:extLst')
 dst.notes_slide.notes_text_frame.text=src.notes_slide.notes_text_frame.text
main.save(OUT/'RelConnector-优化复盘-含四版架构图.pptx')
print('Saved 4 architecture slides, 4 SVG figures, and 23-slide integrated deck.')
