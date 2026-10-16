"""Plain, editable Office PPTX. Generate with python-pptx; fixed report snapshot."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
from pptx.chart.data import CategoryChartData
from pptx.oxml.xmlchemy import OxmlElement

OUT=Path(__file__).resolve().parent
R=Presentation();R.slide_width=Inches(13.333);R.slide_height=Inches(7.5)
BLACK='222222';GREY='666666';LINE='A5A5A5';BLUE='245C94';WHITE='FFFFFF';LIGHT='F0F0F0';PALE='EAF0F7'
FONT='Microsoft YaHei'; NOTES=[]
def rgb(x):return RGBColor.from_string(x)
def plain(shape):
 shape._element.spPr.append(OxmlElement('a:effectLst'))
 return shape

def text(s,x,y,w,h,t,sz=19,b=False,c=BLACK,align=None):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.clear();tf.word_wrap=True
 tf.margin_left=tf.margin_right=Inches(.025);tf.margin_top=tf.margin_bottom=0
 for j,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if j==0 else tf.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(sz);p.font.bold=b;p.font.color.rgb=rgb(c);p.space_after=Pt(4);p.line_spacing=1.10
  if align is not None:p.alignment=align
  for run in p.runs:
   rp=run._r.get_or_add_rPr();ea=OxmlElement('a:ea');ea.set('typeface',FONT);rp.append(ea)
 return sh

def line(s,x1,y1,x2,y2,c=LINE,arrow=False):
 sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=rgb(c);sh.line.width=Pt(1.1);plain(sh)
 if arrow:
  end=OxmlElement('a:tailEnd');end.set('type','triangle');sh._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln').append(end)
 return sh

def rect(s,x,y,w,h,fill=WHITE,stroke=LINE):
 sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
 if stroke:sh.line.color.rgb=rgb(stroke);sh.line.width=Pt(.9)
 else:sh.line.fill.background()
 return plain(sh)

def node(s,x,y,w,h,t,sz=18,fill=WHITE):
 rect(s,x,y,w,h,fill);text(s,x+.08,y+.11,w-.16,h-.15,t,sz,align=PP_ALIGN.CENTER)

def slide(title,source='优化演进报告（2026-09-26 快照）'):
 s=R.slides.add_slide(R.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(WHITE)
 text(s,.48,.28,12.35,.6,title,27,True);line(s,.5,.99,12.83,.99,BLACK)
 text(s,.51,7.18,11.8,.19,source,9,c=GREY);text(s,12.25,7.15,.57,.27,str(len(R.slides)),11,c=GREY,align=PP_ALIGN.RIGHT)
 return s

def heading(s,x,y,w,t):text(s,x,y,w,.37,t,20,True,c=BLUE)
def bullets(s,x,y,w,h,items,sz=18):text(s,x,y,w,h,'\n'.join('• '+t for t in items),sz)
def conclusion(s,t,y=6.62):
 line(s,.52,y-.10,12.8,y-.10);text(s,.55,y,12.22,.43,t,17,True)
def note(s,t,src='optimization-evolution-report-20260926.md'):
 title=s.shapes[0].text;full=t+'\n\n来源：doc/'+src+'\n本页基于原报告快照，未重新执行实验。';s.notes_slide.notes_text_frame.text=full;NOTES.append((len(R.slides),title,full))

def table(s,x,y,widths,headers,rows,rowh=.48,sz=15):
 shape=s.shapes.add_table(len(rows)+1,len(headers),Inches(x),Inches(y),Inches(sum(widths)),Inches(rowh*(len(rows)+1)))
 tab=shape.table
 for j,w in enumerate(widths):tab.columns[j].width=Inches(w)
 for i,row in enumerate([headers]+rows):
  tab.rows[i].height=Inches(rowh)
  for j,val in enumerate(row):
   cell=tab.cell(i,j);cell.text=str(val);cell.margin_left=Inches(.09);cell.margin_right=Inches(.05);cell.margin_top=Inches(.055);cell.margin_bottom=0
   cell.fill.solid();cell.fill.fore_color.rgb=rgb(LIGHT if i==0 else WHITE);cell.vertical_anchor=MSO_ANCHOR.TOP
   tc=cell._tc.get_or_add_tcPr()
   for tag in ['lnL','lnR','lnT','lnB']:
    ln=OxmlElement('a:'+tag);ln.set('w','6350');fill=OxmlElement('a:solidFill');cl=OxmlElement('a:srgbClr');cl.set('val',LINE);fill.append(cl);ln.append(fill);tc.append(ln)
   for p in cell.text_frame.paragraphs:
    p.font.name=FONT;p.font.size=Pt(sz);p.font.bold=i==0;p.font.color.rgb=rgb(BLACK);p.space_after=Pt(0)
    for run in p.runs:
     ea=OxmlElement('a:ea');ea.set('typeface',FONT);run._r.get_or_add_rPr().append(ea)
 return shape

def bars(s,x,y,w,h,categories,values,maximum=None,fmt='0.0',title=None):
 d=CategoryChartData();d.categories=list(reversed(categories));d.add_series('测量值',list(reversed(values)))
 ch=s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x),Inches(y),Inches(w),Inches(h),d).chart
 ch.has_legend=False;ch.has_title=False;ch.chart_style=2
 p=ch.plots[0];p.gap_width=70;p.has_data_labels=True;p.data_labels.position=XL_LABEL_POSITION.OUTSIDE_END;p.data_labels.number_format=fmt;p.data_labels.font.size=Pt(13);p.data_labels.font.name=FONT
 se=p.series[0];se.format.fill.solid();se.format.fill.fore_color.rgb=rgb(BLUE);se.format.line.fill.background()
 va=ch.value_axis;va.minimum_scale=0
 if maximum is not None:va.maximum_scale=maximum
 va.has_major_gridlines=True;va.major_gridlines.format.line.color.rgb=rgb('E1E1E1');va.tick_labels.font.size=Pt(11);va.tick_labels.font.name=FONT;va.tick_labels.number_format=fmt;va.major_tick_mark=XL_TICK_MARK.NONE
 ca=ch.category_axis;ca.tick_labels.font.size=Pt(14);ca.tick_labels.font.name=FONT;ca.major_tick_mark=XL_TICK_MARK.NONE
 if title:heading(s,x,y-.42,w,title)
 return ch

def sequence(s,labels,x=.7,y=2,w=2.5,h=.7,gap=.5,sz=17):
 for i,t in enumerate(labels):
  xx=x+i*(w+gap);node(s,xx,y,w,h,t,sz)
  if i<len(labels)-1:line(s,xx+w,y+h/2,xx+w+gap,y+h/2,BLACK,True)

# 1: no empty title page
s=slide('RelConnector 优化演进：在有限内存下完成关系图训练')
text(s,.56,1.16,12.1,.58,'核心问题：业务特征无法全量放进内存，如何减少在线训练的读取、编码和等待？',21,True)
sequence(s,['任务 seed','常驻图拓扑采样','按需 SQL 特征','组装 → GPU 训练'],x=.7,y=2.03,w=2.6,gap=.5,h=.75)
table(s,.6,3.18,[1.4,3.3,4.1,3.32],['版本','基本方式','优化重点','正式比较角色'],[
 ['vanilla','按采样节点读取特征','已有基础 SQL / raw cache','未引入后续优化的基线'],
 ['v1','在线读取 + 分层复用','去重、文本 / encoded 缓存','单进程优化版本'],
 ['v2','采样 / 组装 / 训练分进程','并行生产、共享与磁盘缓存','当前最新实现'],
 ['batch','训练前全量特征物化','传统训练方式的背景说明','不纳入对比实验']],.56,16)
bullets(s,.64,6.13,12.0,.38,['系统仍需将图拓扑放入内存；目标是降低全量业务特征的常驻需求。'],17)
note(s,'本文使用2026-09-26详细报告快照。batch不做性能排名。vanilla已有混合SQL读取、raw cache、线程执行器，未优化是相对后续系统优化。标题页兼作研究范围说明。')
# 2
s=slide('1. 优化路线：减少重复工作 → 提高并行度 → 降低管理成本')
for y,label,ls,caption in [
 (1.55,'vanilla',['采样','SQL + 解码','文本 + Frame','训练'],'逐 batch 准备；同一节点和文本可能被反复处理。'),
 (3.02,'v1',['采样 + 窗口','合并读取 / raw','text / encoded','按序训练'],'把相邻请求合并，只处理 miss；SQL 与 encode 线程重叠。'),
 (4.49,'v2',['采样进程','有界 plan 队列','组装进程池','GPU 主进程'],'独立进程供给特征；共享结果，按需复用磁盘文本向量。')]:
 text(s,.57,y+.10,1.25,.5,label,21,True);sequence(s,ls,x=2,y=y,w=2.23,h=.67,gap=.48,sz=16);text(s,2,y+.78,10.7,.38,caption,17)
conclusion(s,'后续 v2 修复关注：HM 的传输粒度、Salt/Stack 的共享查询、mmap 与 WAL 资源增长。')
note(s,'历史v1表代表早期缓存版本，不等于当前relconnector目录完整实现。cache_baseline是早期v1冻结版本，不额外作为主对照。v2有独立命名空间，当前工作区并非固定发布tag。')
# 3
s=slide('2. 瓶颈定位：GPU 主要在等待 CPU 特征准备', '来源：bottleneck-analysis-and-optimization.md｜历史数据，用于解释优化依据')
bars(s,.6,1.83,5.55,3.45,['Amazon','Arxiv','Stack'],[.973,.890,.865],1.15,'0.0%',title='历史 ready 等待占训练时间的比例')
heading(s,6.55,1.38,6,'特征路径为何昂贵？')
sequence(s,['SQL / 解码','文本编码','Frame 组装'],x=6.56,y=2.02,w=1.76,gap=.28,h=.7,sz=15)
bullets(s,6.55,3.04,6.03,2.58,[
 '同一节点被反复采样，重复读取和转换。',
 '文本向量未复用时，GloVe 被反复调用。',
 '命中缓存后，仍有查询、复制和展开。',
 '单个特征 worker 可能无法持续供给 GPU。'],19)
text(s,.7,5.55,11.95,.66,'稳态速度由最慢的阶段决定：采样供给、特征生产、GPU 更新。\n初始化、流水线填充与退出，仍会影响端到端时间。',19)
conclusion(s,'先看 wall time 和 ready 等待，再看阶段计时；并行 / 嵌套操作耗时不能直接相加。')
note(s,'97.3%、89.0%、86.5%来自历史版本；不代表当前v2。不能简单将多个worker服务时间相加解释训练wall。','bottleneck-analysis-and-optimization.md')
# 4
s=slide('3. SQL 优化：把扫描变成按 ID 查找，避免过量读取')
heading(s,.61,1.28,5.95,'① rowid 查询：转换 ID，不改索引键')
node(s,.64,1.88,5.75,.96,'旧：WHERE rowid − 1 IN (...)\n表达式可能破坏直接 rowid 查找',18)
line(s,3.52,2.84,3.52,3.14,BLACK,True)
node(s,.64,3.14,5.75,1.16,'新：WHERE rowid IN (node_id + 1, ...)\nORDER BY rowid\nSELECT 输出时再使用 rowid − 1',17,PALE)
heading(s,6.77,1.28,5.91,'② 稀疏 / 稠密访问使用不同读取方式')
text(s,6.81,1.94,5.72,.39,'稀疏请求：只取零散 ID，合并 IN 查询',18)
for i in range(10):
 rect(s,6.83+i*.53,2.55,.39,.42,PALE if i in [0,4,9] else WHITE);text(s,6.84+i*.53,2.59,.37,.3,str(i),13,align=PP_ALIGN.CENTER)
text(s,6.81,3.15,5.72,.43,'稠密请求：读取整块，进入 raw cache',18)
for i in range(10):rect(s,6.83+i*.53,3.77,.39,.42,PALE if 2<=i<=7 else WHITE)
bullets(s,.66,4.79,12,1.45,['SQL 放大率关注实际查询行数 / 实际需要查询的行数；featureless 节点须排除。','raw 命中省下数据库读取和解码，但并不省掉后面的文本 / TensorFrame 工作。','当前 vanilla 已回补 rowid 修复；旧版超时不能代表今天的 vanilla。'],18)
conclusion(s,'收益来自“少扫、少取、少解码”；正式实验需固定数据库映射与修订后的输入语义。')
note(s,'稀疏/稠密混合读取是现有基础能力，不能全部归为v1新增。后续还支持显式连续node ID。示意格子不是实际测量。','feature-pipeline-optimization-plan.md；cache-baseline-correctness-report.md')
# 5
s=slide('4. 去重与窗口：只做一次特征生产，再恢复训练输入')
heading(s,.61,1.27,12,'示例：两个 batch 合并为一个特征窗口（示意数据）')
node(s,.67,1.94,3.14,.78,'Batch A：7, 3, 7, 8',19)
node(s,.67,3.05,3.14,.78,'Batch B：3, 9, 8, 9',19)
node(s,4.64,2.45,3.25,.9,'唯一 ID：3, 7, 8, 9\n读取 + 编码 4 行',18,PALE)
line(s,3.81,2.33,4.64,2.9,BLACK,True);line(s,3.81,3.44,4.64,2.9,BLACK,True)
node(s,9.02,1.94,3.63,.78,'恢复 A：f7, f3, f7, f8',18)
node(s,9.02,3.05,3.63,.78,'恢复 B：f3, f9, f8, f9',18)
line(s,7.89,2.9,9.02,2.33,BLACK,True);line(s,7.89,2.9,9.02,3.44,BLACK,True)
text(s,4.54,3.66,3.75,.6,'8 个请求 → 4 行生产\n训练仍保留 8 个位置',17,True,BLUE,PP_ALIGN.CENTER)
table(s,.66,4.64,[2.2,4.75,5.14],['复用范围','省下的工作','必须保留的语义'],[
 ['批内 / 推荐分支','合并同表节点及推荐任务分支','节点重复位置、边顺序、label'],
 ['相邻 batch 窗口','合并 SQL、解码与静态特征编码','分别恢复各 batch 和 seed time']],.6,16)
conclusion(s,'窗口越大不一定越好：受 batch / byte 上限约束；跨 epoch 预取仍须拆成独立窗口。')
note(s,'图示数字为教学例子，不是性能实验。历史固定window4batch/512MiB，动态窗口后按复用率和缓存覆盖调整。去重不改变训练任务样本数。')
# 6
s=slide('5. 缓存逐层上移：越接近最终特征，省掉的重复工作越多')
sequence(s,['SQL → 解码','DataFrame','GloVe 向量','TensorFrame'],x=.72,y=1.65,w=2.56,gap=.48,h=.7)
for x,t in [(3.76,'raw cache'),(6.8,'text cache'),(9.84,'encoded cache')]:
 line(s,x+1.28,2.35,x+1.28,2.74,BLACK,True);node(s,x,2.75,2.56,.58,t,18,PALE)
table(s,.63,3.60,[1.85,3.26,3.19,3.85],['层级','缓存对象','命中可省掉','命中后仍需付出的成本'],[
 ['raw','解码后的 DataFrame 块','SQL / 解码','文本处理、Frame 转换'],
 ['text','冻结的 300 维句向量','GloVe 模型执行','查询、gather、Frame 转换'],
 ['encoded','静态 CPU TensorFrame','重复 prepare / converter','缓存查询、切片与恢复顺序']],.57,16)
bullets(s,.68,6.03,12,.42,['按实际持有的 storage 计费；有界淘汰与准入保护热点，超大项拒绝入缓存。'],17)
conclusion(s,'高命中率不等于低成本；后续 HM、Salt、Stack 都说明“访问缓存本身”可能成为瓶颈。')
note(s,'encoded以(table,node_id)索引，整块持有但仅请求miss编码。不能缓存GNN trainable hidden states。raw节点常量、text和encoded的开销层级不同。','adaptive-feature-optimization-report.md')
# 7
s=slide('6. v1 第二轮：减少通用框架开销，并让 SQL 与编码重叠')
heading(s,.64,1.27,5.95,'文本快路径：保留官方计算语义')
sequence(s,['分词','词向量查表','平均池化'],x=.65,y=1.93,w=1.78,gap=.22,h=.66,sz=14)
bullets(s,.65,2.92,5.88,1.33,['direct 复用官方模块；不匹配时回退 encode。','连续 slab 减少小 Tensor / storage 对象。'],18)
heading(s,6.85,1.27,5.8,'线程流水线：按顺序消费')
sequence(s,['SQL fetch','有界 fetched','1/2 encode'],x=6.85,y=1.93,w=1.78,gap=.22,h=.66,sz=15)
bullets(s,6.85,2.92,5.77,1.33,['窗口按重复率与 raw 命中率自适应。','schema / topology 持久化降低重复初始化。'],18)
table(s,.64,4.47,[3.68,2.32,2.36,3.81],['已有实验','对照','优化','解释范围'],[
 ['文本微基准（20 轮）','0.2123 s','0.1334 s','局部约 1.59×，非训练加速'],
 ['Salt GPU 2×50 batch','213.58 s','140.99 s','组合优化，训练耗时 ↓34.0%'],
 ['Ratebeer GPU 2×20','57.77 s','52.83 s','短程训练耗时 ↓8.5%']],.46,15)
conclusion(s,'F1 CPU 小任务反而退化约 1.5%；多 worker 的收益取决于特征瓶颈和额外开销。')
note(s,'direct固定文本与official逐值一致，但CUDA短回归loss未全面达到严格阈值。Salt/Ratebeer组合修改无法拆成单项贡献。表中训练时间不是全流程wall。','adaptive-feature-optimization-report.md')
# 8
s=slide('7. 正确性：先对齐输入，再比较性能')
sequence(s,['同一数据 / stype','同一采样计划','同一 TensorFrame','loss / 参数更新'],x=.7,y=1.59,w=2.57,gap=.48,h=.75,sz=17)
table(s,.63,2.77,[2.05,5.05,5.03],['必须固定','具体要求','为什么影响比较'],[
 ['字段与目标','共同读取官方 stype；按任务隐藏目标列','改型 / 删列会改变编码量和学习问题'],
 ['结构与时间','节点 / 边顺序、重复项、seed time','去重只省生产，不能减少训练输入'],
 ['缓存内容','只保存静态特征与冻结文本向量','训练中的 GNN 表示会随参数更新'],
 ['执行顺序','strict 固定顺序；ready 单独评估','batch 顺序改变优化器轨迹']],.64,16)
bullets(s,.68,6.03,12,.4,['已有证据：固定 fixture 逐值一致；真实 F1 推荐任务的 60 组特征与结构一致。'],17)
conclusion(s,'修订标识 relbench-correctness-v1；修复前后成绩分开，不能把历史 v1 当成当前 v1。')
note(s,'autocomplete目标隐藏、Ratebeer external勘误、官方随机stype抽样与dtype还原均回补。四版本fixture通过不等于全任务CUDA与ready都逐位一致。','cache-baseline-correctness-report.md')
# 9
s=slide('8. v2 多进程架构：组装进程完成“读取到特征”的完整路径')
node(s,.61,2.45,1.8,.98,'采样进程\nseed + 拓扑',18)
node(s,2.94,2.45,1.8,.98,'plan 队列\n有界窗口',18)
line(s,2.41,2.94,2.94,2.94,BLACK,True)
for y,label in [(1.65,'Assembly 1'),(2.83,'Assembly 2'),(4.01,'Assembly N')]:
 node(s,5.37,y,3.0,.85,label+'\nSQL → GloVe → Frame',16)
 line(s,4.74,2.94,5.37,y+.425,BLACK,True)
node(s,9.0,2.45,1.5,.98,'ready 队列\nprepared',17)
for y in [1.65,2.83,4.01]:line(s,8.37,y+.425,9.0,2.94,BLACK,True)
node(s,11.02,2.45,1.68,.98,'GPU 主进程\n训练更新',18);line(s,10.5,2.94,11.02,2.94,BLACK,True)
node(s,.64,4.76,4.02,.75,'常驻共享拓扑 / SQLite 数据源',17,PALE)
node(s,9.0,4.76,3.7,.75,'共享特征缓存 + 磁盘文本缓存',17,PALE)
line(s,4.66,5.14,5.37,4.44,LINE,True);line(s,9.0,5.14,8.37,4.44,LINE,True)
bullets(s,.69,5.93,12.03,.61,['worker 内保留 reader、原始字符串和 DataFrame；跨进程传计划与准备好的 tensor。'],18)
conclusion(s,'v1 单进程优化后 assemble 仍受限；v2 将采样、组装与训练分进程重叠执行。')
note(s,'图为概念架构，Assembly N指可配置池，默认4。图中的SQLite/拓扑来源合并示意，实际采样使用共享拓扑，worker自行读取SQLite。shared encoded索引由manager管理。','../relconnector_v2/relconnector_v2/runtime/process_pipeline.py')
# 10
s=slide('9. 调度：ready 减少队头等待，但训练顺序会改变')
heading(s,.65,1.25,12,'时序示意：B2 在 t=1 就绪，B3 在 t=2 就绪，B1 在 t=4 就绪')
# illustrative timeline, explicitly not measurement
x0=2.18;step=1.25
for t in range(8):text(s,x0+t*step-.11,1.84,.5,.3,str(t),14,align=PP_ALIGN.CENTER)
text(s,12.0,1.84,.8,.3,'时间',14)
for yy,lab in [(2.44,'strict'),(3.64,'ready')]:
 text(s,.75,yy+.1,1.2,.4,lab,21,True);line(s,x0,yy+.65,12.28,yy+.65,LINE)
# strict wait and three train blocks
rect(s,x0,2.44,step*4,.62,LIGHT);text(s,x0+.1,2.56,step*4-.2,.32,'等待 B1（B2 / B3 已就绪）',16,align=PP_ALIGN.CENTER)
for t,l in [(4,'B1'),(5,'B2'),(6,'B3')]:node(s,x0+t*step,2.44,step,.62,l,17,PALE)
for t,l in [(1,'B2'),(2,'B3'),(4,'B1')]:node(s,x0+t*step,3.64,step,.62,l,17,PALE)
rect(s,x0+3*step,3.64,step,.62,LIGHT);text(s,x0+3*step,3.76,step,.35,'等待',15,align=PP_ALIGN.CENTER)
table(s,.66,4.55,[2.0,5.0,5.1],['机制','作用','边界'],[
 ['epoch gate','只训练当前 epoch；下一轮可提前准备','一个特征窗口不能混合多个 epoch'],
 ['队列 + inflight','限制排队字节与未完成窗口','超前准备不应无限累积'],
 ['动态活跃数','ready 饥饿时扩展，高水位时减少','预先 spawn；暂停不等于释放内存']],.47,15)
conclusion(s,'strict 用于固定顺序的正确性 / 性能对照；ready 的吞吐收益与收敛影响单独报告。')
note(s,'图为示意，假设单batch训练1时间单位，无真实测量意义。ready示例0到1空闲、3到4等待。当前epoch全部完成后才训练下一轮。动态控制是进程许可，不是按需spawn回收。')
# 11
s=slide('10. 文本缓存路径：先复用，再编码；高复用时绕过共享锁')
sequence(s,['规范化文本','本地槽位索引','共享文本缓存','磁盘向量缓存'],x=.67,y=1.69,w=2.57,gap=.47,h=.72,sz=17)
# hit outcomes below each layer
for x,lab in [(3.71,'命中：直接 gather'),(6.75,'命中：取已发布向量'),(9.79,'命中：读取分片')]:
 line(s,x+1.285,2.41,x+1.285,2.83,BLACK,True);node(s,x,2.84,2.57,.66,lab,16,PALE)
text(s,3.39,1.15,9.2,.38,'向右表示 miss；本地索引只在高收益工作负载下激活。',17,c=GREY)
node(s,9.79,4.06,2.57,.7,'均 miss → GloVe 编码',16)
line(s,12.36,2.05,12.76,2.05,LINE);line(s,12.76,2.05,12.76,4.41,LINE);line(s,12.76,4.41,12.36,4.41,LINE,True)
text(s,11.74,3.62,.95,.28,'磁盘 miss',12,c=GREY)
text(s,.7,4.02,8.38,1.58,'• 共享 reservation 协调同一 miss，避免多个 worker 同时编码。\n• 本地只记“文本 → 不可变共享槽位”，不复制整份向量。\n• 低复用时释放索引并冷却；每 worker 约 32 MiB 元数据预算。',18)
text(s,.7,5.8,12.0,.56,'磁盘 adaptive：数据库 ≥256 MiB，模型输入 ≥4,096、模型耗时 ≥1 s，内存命中率 ≤10%。',17)
conclusion(s,'索引策略与磁盘策略独立；F1 可不激活，Arxiv 低复用时可自动退出本地索引。')
note(s,'这是逻辑查找顺序示意，actual local-index/shared-cache/disk reservations需要协调；向下命中到向量输出，右下GloVe仅表示所有层未命中，不能把命中向量再编码。缓存发布回共享/磁盘省略回流线。','v2-salt-regression-20260926.md；v2-disk-text-cache-experiment-20260925.md')
# 12
s=slide('11. HM / Salt：命中缓存以后，访问成本才是瓶颈')
heading(s,.64,1.26,5.99,'HM：整块 RPC → 只传行 / 本地缓存')
heading(s,6.8,1.26,5.95,'Salt：频繁共享查询 → 本地槽位索引')
sequence(s,['少量请求','整块传输'],x=.66,y=1.87,w=2.52,gap=.59,h=.6,sz=17)
sequence(s,['重复文本','反复取共享锁'],x=6.81,y=1.87,w=2.52,gap=.59,h=.6,sz=17)
bullets(s,.67,2.75,5.77,1.35,['article 命中率 99.77%，但 RPC 很重。','先只返回命中行，再提供 local encoded。'],18)
bullets(s,6.82,2.75,5.77,1.35,['共享命中率约 99.93%，第二轮编码为 0。','向量共享不变；重复命中绕过共享锁。'],18)
bars(s,.68,4.05,5.61,1.73,['原 v2','修复后'],[4603.0,1756.3],5700,'0',title='HM：两轮 pipeline（秒），2.62×')
bars(s,6.83,4.05,5.61,1.73,['原 v2','修复后'],[18260.23,5208.23],22500,'0',title='Salt：两轮 wall（秒），3.51×')
text(s,.73,5.98,5.72,.46,'shared 2 GiB → local 总计 4 GiB',16,c=GREY)
text(s,6.87,5.98,5.72,.46,'完整 6,340 batch，模型输入不变',16,c=GREY)
conclusion(s,'这些是“修复旧 v2”的倍数；HM 还改变作用域 / 容量，不能写成对 v1 的单因素提升。')
note(s,'HM等预算短测：206.0→183.5→157.7s支持RPC归因；完整结果改变容量。HM wall2.87x还包含初始化暖。Salt修复包含位置索引和共享查询批量化，相对历史v1仅1.088x。','v2-hm-regression-20260925.md；v2-salt-regression-20260926.md')
# 13
s=slide('12. Stack / Amazon：并行训练还需要控制系统资源')
heading(s,.64,1.28,5.85,'Stack：查询竞争与 mmap 数持续增长')
sequence(s,['逐槽查询 / clone','长锁等待'],x=.64,y=1.9,w=2.55,gap=.52,h=.62,sz=16)
sequence(s,['分片映射不关闭','映射 / FD 耗尽'],x=.64,y=2.86,w=2.55,gap=.52,h=.62,sz=16)
bullets(s,.66,3.9,5.78,1.53,['哈希与批量 gather 移到锁外。','每 cache 最多 1,024 个映射，LRU 回收。','同步错误通道，避免报错时再建线程。'],18)
heading(s,6.83,1.28,5.85,'Amazon：WAL 增长与 checkpoint 抖动')
sequence(s,['大量小写入','WAL 约 60.46 GB'],x=6.83,y=1.9,w=2.55,gap=.52,h=.62,sz=16)
sequence(s,['固定 64 MiB 阈值','频繁刷盘 / 等锁'],x=6.83,y=2.86,w=2.55,gap=.52,h=.62,sz=16)
bullets(s,6.86,3.9,5.77,1.53,['每 writer 独占 8,192 行追加分片。','按索引大小与剩余空间设置回收阈值。','阈值是触发条件，不是 WAL 硬上限。'],18)
text(s,.67,5.78,5.8,.62,'完整 13,228 batch：3,665.66 s\n越过旧挂起点，worker 正常退出。',17,True)
text(s,6.86,5.78,5.8,.62,'2×100 batch：676.01 s（仅短测）\nWAL 峰值约 1.35 GiB，退出归零。',17,True)
conclusion(s,'内存有界还不够：文件数、映射数、日志增长与错误传输都影响能否持续完成训练。')
note(s,'Stack完整结果暖磁盘；旧早期停滞缺调用栈不能武断说都是死锁。后续mmap资源链有现场/隔离复现。Amazon续跑已经合并6584936条向量，不能当冷启动。','v2-stack-regression-20260926.md；v2-amazon-regression-20260926.md')
# 14
s=slide('13. 磁盘缓存：首轮构建成本与后续复用收益分开看')
bars(s,.6,1.96,5.87,3.5,['cold','off','warm'],[396.6,204.7,190.5],480,'0.0',title='Arxiv 2×50 batch：wall time（秒）')
heading(s,6.93,1.28,5.7,'缓存生命周期')
sequence(s,['首次遇到文本','编码 → 写入'],x=6.93,y=1.93,w=2.5,gap=.55,h=.68,sz=17)
sequence(s,['后续 batch / epoch','索引 → 向量'],x=6.93,y=3.08,w=2.5,gap=.55,h=.68,sz=16)
bullets(s,6.94,4.13,5.7,1.76,['warm 比 off 快约 6.9%。','模型输入从 574,119 降到 0。','总时间仍受转换、查询、gather 限制。'],18)
text(s,.72,5.72,11.93,.66,'冷启动 396.6 s 包含首次初始化和写入；不能把 cold / warm 的 2.08× 全算成编码收益。\n100 万条 300 维 float32 向量约 1.12 GiB，仅为向量 payload，不含索引和 WAL。',17)
conclusion(s,'mmap LRU 只限制打开的映射，不限制持久化向量总量；需要记录初始 / 最终磁盘占用。')
note(s,'两个Assembly worker、ready顺序，cold是新的embedding缓存并含首次初始化。磁盘激活并不保证每任务更快，需要多轮摊销。','v2-disk-text-cache-experiment-20260925.md')
# 15
s=slide('14. 结果总览：8 项成功记录，仍有退化和未完成任务')
text(s,.64,1.19,12.05,.42,'以下为报告快照中的最新成功记录；历史 v1 为单轮，v2 为两轮。',18)
table(s,.62,1.95,[1.13,1.32,1.53,1.53],['任务','batch 数','wall / s','s / batch'],[
 ['Arxiv','2,088','981.64','0.47013'],['Avito','8,644','1,604.83','0.18566'],['Event','76','2,193.96','28.86783'],['F1','12','26.57','2.21388'],['HM','14,972','1,760.13','0.11756'],['Salt','6,340','5,208.23','0.82149'],['Stack','13,228','3,665.66','0.27711'],['Trial','918','656.95','0.71563']],.43,14)
bars(s,6.73,2.01,5.7,3.96,['Arxiv','Stack','HM','Avito','F1','Salt','Event','Trial'],[5.817,3.508,1.981,1.490,1.197,1.088,.881,.792],7,'0.000',title='历史 v1 / v2 单位 batch 耗时比')
text(s,6.89,6.01,5.56,.37,'比值 >1：此组记录中 v2 更快。',16,True)
text(s,.68,6.08,5.58,.41,'Amazon、Ratebeer：完整结果待补。',16,True)
conclusion(s,'硬件、正确性版本、冷热缓存及顺序不同：仅用于观察趋势，不是严格配对加速结论。')
note(s,'数据见报告§8；历史A30，v2文档RTX4090，HM local4GiB，其他多shared2GiB；各任务并非最终同版本统一重跑。Event初始化主导；Trial仍有ready等待。Ratebeer矩阵上限每轮10000，而历史完整20743，不能当完整epoch。')
# 16
s=slide('15. 内存目标是否达成？必须统计整个进程树')
heading(s,.65,1.28,12,'当前 peak_rss_mb = RUSAGE_SELF，只记录训练主进程，不能代表系统总内存。')
node(s,.68,2.06,2.62,.78,'训练主进程\n当前 RSS 指标',18,PALE)
node(s,3.82,2.06,2.6,.78,'采样进程\n拓扑 / 临时数组',18)
node(s,6.96,2.06,2.6,.78,'Assembly × N\n模型 / Frame / SQL',18)
node(s,10.1,2.06,2.58,.78,'缓存 manager\n索引 / 共享数据',18)
line(s,1.99,2.84,1.99,3.28);line(s,5.12,2.84,5.12,3.28);line(s,8.26,2.84,8.26,3.28);line(s,11.39,2.84,11.39,3.28);line(s,1.99,3.28,11.39,3.28)
text(s,2.36,3.49,9.62,.49,'一起测量：进程树 PSS / USS、共享内存和 cgroup 峰值',19,True,align=PP_ALIGN.CENTER)
table(s,.68,4.36,[3.0,3.4,5.64],['配置预算（不是实测）','当前典型值','解释'],[
 ['raw / encoded / text','4 GiB / 2 GiB / 512 MiB','本地缓存按 worker 分摊，共享只计总预算'],
 ['plan / ready 队列','2 GiB / 16 GiB','容量不等于实际峰值，也非系统硬上限'],
 ['其他成本','模型、临时对象、在途数据','暂停 worker 不等于归还其全部内存']],.53,15)
conclusion(s,'不能由主进程 2–4 GiB 宣称整个 v2 低内存达标；还需显存、磁盘、FD / mmap 峰值。')
note(s,'各进程RSS相加可能重复计算共享页。拓扑仍常驻；构建临时数组和操作系统余量需要纳入。HM encoded例外local总4GiB。默认队列16GiB不能当作适合任意16/32GiB机器的安全配置。')
# 17
s=slide('16. 下一轮实验：统一约束，拆开验证每项优化')
sequence(s,['固定源码 / 数据','固定内存与硬件','分别测冷 / 热','记录完整工作量'],x=.69,y=1.54,w=2.56,gap=.48,h=.72,sz=17)
table(s,.66,2.55,[2.34,4.0,5.73],['对照维度','实验设置','回答的问题 / 主要指标'],[
 ['主版本','vanilla / v1 / v2','相同语义与总内存限制下，能否完成、吞吐多少'],
 ['并行度','1 / 2 / 4 worker，等总预算','ready 等待、CPU/I/O 饱和度、进程树内存'],
 ['encoded 缓存','off / shared / local，等容量','省编码收益能否覆盖 RPC / 复制成本'],
 ['调度 / 文本索引','strict vs ready；shared vs adaptive','顺序对收敛的影响；共享查询成本'],
 ['磁盘 / 初始化','全冷；初始化暖文本冷；全热','首次成本、多轮摊销、磁盘增长']],.57,15)
text(s,.7,6.03,12,.38,'重复运行并报告波动；短测与完整 epoch 分开，初始缓存不得在后测中默默继承。',17)
conclusion(s,'优先补 Amazon / Ratebeer；诊断 Event / Trial，再用统一最终代码重跑已完成任务。')
note(s,'主对照排除batch。缓存容量可不同，但必须受统一总内存限额，消融则按等预算/容量。HM目前encoded配置按数据集特配，需要披露，未来可以转为运行时策略。')
# 18
s=slide('17. 总结：每一轮优化都在解决新的关键路径问题')
table(s,.66,1.45,[1.58,3.53,3.28,3.68],['阶段','主要改动','解决的问题','仍需关注'],[
 ['vanilla','按需 SQL 读取与在线组装','避免全量业务特征驻留','重复读取 / 编码，CPU 供给不足'],
 ['v1 第一轮','rowid 修复、去重、窗口、text','减少扫描与重复特征生产','大文本工作集导致缓存抖动'],
 ['v1 第二轮','encoded、direct、slab、线程','减少重复转换与调度开销','小任务并行开销，线程并行限制'],
 ['v2','多进程、有界队列、共享 / 磁盘','提高供给与跨轮次复用','RPC、共享锁与长期资源增长'],
 ['v2 后续修复','只传行、槽位索引、mmap/WAL','降低命中成本，保障持续运行','统一最终版本性能与总内存验收']],.68,15)
heading(s,.7,5.86,12,'最重要的认识：缓存“命中”只是中间指标，最终要看训练是否更快、资源是否有界。')
conclusion(s,'当前证据支持优化方向；正式结论依赖同语义、同硬件、同内存限额下的配对实验。')
note(s,'主来源doc/optimization-evolution-report-20260926.md。其他文档：gpu-feature-optimization-experiment.md、adaptive-feature-optimization-report.md、cache-baseline-correctness-report.md、v2各任务回归报告。原始v2结果benchmarks/v2-full-matrix-20260925/v2.jsonl。此版18页，白底黑字，简单原生流程图、表格和可编辑Office图表，无卡片、阴影、装饰页。')

assert len(R.slides)==18
for i,s in enumerate(R.slides,1):
 for sh in s.shapes:
  for effect in sh._element.xpath('.//a:effectRef'): effect.set('idx','0')
  assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=R.slide_width+30 and sh.top+sh.height<=R.slide_height+30,(i,sh.name)
R.core_properties.title='RelConnector 优化演进（简洁图解版）';R.core_properties.author='RelConnector';R.core_properties.subject='有限内存训练：vanilla → v1 → v2'
R.save(OUT/'RelConnector-优化演进-简洁图解版.pptx')
(OUT/'speaker-notes.md').write_text('# 逐页讲解备注\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{body}' for i,title,body in NOTES),encoding='utf-8')
(OUT/'README.md').write_text('''# RelConnector 优化演进：简洁图解版

18 页 Office 标准 PPTX。白底黑字、少量蓝色；去掉装饰性封面、卡片、圆角、渐变与阴影。文字、流程框、箭头、表格和图表均为原生可编辑对象，图表内含数据表。

- [PowerPoint 文件](RelConnector-优化演进-简洁图解版.pptx)
- [PDF 预览](RelConnector-优化演进-简洁图解版.pdf)
- [逐页备注](speaker-notes.md)

字体使用 Office 常见的 Microsoft YaHei（微软雅黑）。PDF 预览使用 Noto Sans CJK SC 替代渲染。图示时序 / ID 示例如有虚构数据，页内明确标为示意；实验数值固定于详细报告的 2026-09-26 快照，没有重新运行训练。batch 不纳入正式比较。

重建：安装 `python-pptx`，运行 `python build_slides.py`。PDF 由 LibreOffice 导出。
''',encoding='utf-8')
print('Saved',len(R.slides),'slides;',sum(1 for s in R.slides for sh in s.shapes if sh.has_chart),'editable charts')
