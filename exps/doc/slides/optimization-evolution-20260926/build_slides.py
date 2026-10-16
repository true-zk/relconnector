"""Generate editable Chinese slides. Requires python-pptx; no training is run."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
import json

OUT=Path(__file__).resolve().parent
R=Presentation(); R.slide_width=Inches(13.333); R.slide_height=Inches(7.5)
BG='F5F7FB'; WHITE='FFFFFF'; NAVY='12263A'; INK='203449'; MUTED='66788A'; TEAL='008F87'; BLUE='4476CB'; ORANGE='D88731'; RED='B84D53'; LIGHT='E7EEF5'; PALE='E4F4F0'; FONT='Noto Sans CJK SC'
slides=[]; notes=[]
def color(c):return RGBColor.from_string(c)
def box(s,x,y,w,h,fill=WHITE,line=None,radius=False):
 q=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
 q.fill.solid();q.fill.fore_color.rgb=color(fill)
 q.line.fill.background() if line is None else None
 if line: q.line.color.rgb=color(line)
 if radius:
  try:q.adjustments[0]=.08
  except Exception:pass
 return q

def txt(s,x,y,w,h,t,size=18,c=INK,bold=False,align=None):
 q=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h));tf=q.text_frame;tf.clear();tf.word_wrap=True
 tf.margin_left=tf.margin_right=Inches(.015);tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(c);p.space_after=Pt(6);p.space_before=Pt(0)
  if align is not None:p.alignment=align
  for run in p.runs:
   rp=run._r.get_or_add_rPr();ea=OxmlElement('a:ea');ea.set('typeface',FONT);rp.append(ea)
 return q

def slide(title,kicker='RELCONNECTOR / 优化演进',sub='',source='',dark=False):
 s=R.slides.add_slide(R.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(NAVY if dark else BG);slides.append(s)
 fg=WHITE if dark else INK
 box(s,.5,.42,.1,.25,TEAL);txt(s,.73,.38,11,.3,kicker,10,TEAL,True)
 txt(s,.7,.94,12, .65,title,28,fg,True)
 if sub:txt(s,.73,1.69,11.85,.55,sub,14,'BCD0DF' if dark else MUTED)
 box(s,.73,7.03,11.87,.012,'3B5064' if dark else 'D7E0E8')
 txt(s,.75,7.12,11.25,.22,source or '资料快照：2026-09-26｜基于优化演进报告；未重新运行训练',9,'9CB0C2' if dark else MUTED)
 txt(s,12.05,7.10,.5,.26,f'{len(slides):02}',11,'9CB0C2' if dark else MUTED,align=PP_ALIGN.RIGHT)
 return s

def note(s,t,refs='optimization-evolution-report-20260926.md'):
 full=t+'\n\n来源（项目 doc/）：'+refs+'\n数据为报告快照，不是本次重跑结果。'
 s.notes_slide.notes_text_frame.text=full;notes.append((len(slides),s.shapes[2].text if s.shapes[2].has_text_frame else '',full))
def takeaway(s,t,y=6.25):
 box(s,.73,y,11.87,.52,PALE,radius=True);txt(s,.94,y+.10,11.4,.34,t,16,TEAL,True)
def card(s,x,y,w,h,label,body,accent=TEAL):
 box(s,x,y,w,h,WHITE,radius=True);box(s,x,y,.055,h,accent);txt(s,x+.23,y+.2,w-.45,.53,label,21,accent,True);txt(s,x+.23,y+.88,w-.45,h-1.02,body,17)
def arrow(s,x,y,w=.43):
 a=s.shapes.add_shape(MSO_SHAPE.CHEVRON,Inches(x),Inches(y),Inches(w),Inches(.34));a.fill.solid();a.fill.fore_color.rgb=color(TEAL);a.line.fill.background()
def flow(s,items,y=3,w=2.55,h=1.3):
 gap=.5;x=.73
 for i,(label,body) in enumerate(items):
  box(s,x,y,w,h,WHITE,radius=True);txt(s,x+.13,y+.19,w-.26,.4,label,19,TEAL,True,PP_ALIGN.CENTER);txt(s,x+.13,y+.72,w-.26,.4,body,13,MUTED,align=PP_ALIGN.CENTER)
  if i<len(items)-1:arrow(s,x+w+.05,y+.48,.37)
  x+=w+gap

def paired(s,rows,maxv,x=.95,y=2.6,w=7.1,unit='s'):
 for i,(label,a,b) in enumerate(rows):
  yy=y+i*1.25;txt(s,x,yy,1.7,.4,label,17,INK,True)
  for j,(v,c) in enumerate([(a,'AAB9C8'),(b,TEAL)]):
   bw=w*v/maxv;box(s,x+1.8,yy+j*.38,max(bw,.035),.24,c);txt(s,x+1.9+bw,yy-.05+j*.38,1.7,.33,f'{v:,.2f} {unit}',13,MUTED)

def table(s,headers,rows,widths,y=2.35,rowh=.46,size=14):
 x=.73
 for j,(h,w) in enumerate(zip(headers,widths)):
  box(s,x,y,w,.5,NAVY);txt(s,x+.1,y+.11,w-.2,.32,h,size,WHITE,True);x+=w
 for i,row in enumerate(rows):
  x=.73
  for j,(v,w) in enumerate(zip(row,widths)):
   box(s,x,y+.5+i*rowh,w,rowh,WHITE if i%2==0 else LIGHT);txt(s,x+.1,y+.59+i*rowh,w-.2,rowh-.07,str(v),size,TEAL if j==len(row)-1 else INK,j==0);x+=w

# 01
s=slide('有限内存下的关系图训练',sub='RelConnector 系统优化演进｜vanilla → v1 → v2',dark=True)
txt(s,.75,2.65,11.7,1.3,'从减少重复工作\n到提高特征供给能力',38,WHITE,True)
for x,n,t in [( .78,'01','按需读取'),(4.85,'02','分层复用'),(8.95,'03','有界并行')]:
 txt(s,x,4.62,1,.6,n,32,TEAL,True);txt(s,x+1.05,4.82,2.8,.45,t,21,WHITE)
txt(s,.78,6.12,11.5,.5,'中文技术汇报  ·  2026.09.26  ·  数据截至报告快照',15,'BCD0DF')
note(s,'本演示围绕有限内存下在线训练的优化机制与证据展开。主对照为 vanilla、v1、v2。batch 仅解释传统全量特征物化方式，不计入比较实验。建议讲解时长20–30分钟。')
# 02
s=slide('研究问题：特征放不进内存，训练如何持续推进？',sub='目标是降低全量业务特征的常驻需求，同时保持训练输入语义。')
card(s,.73,2.55,5.73,2.7,'问题约束','数据集业务特征无法一次载入 RAM\n图拓扑仍常驻内存\n训练节点特征按采样结果读取')
card(s,6.7,2.55,5.9,2.7,'正式比较范围','vanilla：未引入后续优化的在线基线\nv1：单进程优化实现\nv2：当前最新多进程实现',BLUE)
takeaway(s,'batch 仅说明传统全量物化方式，不进入性能排名和加速比统计。')
note(s,'图拓扑依然需要放进内存，因此本项目不是任意规模图的完全外存训练。vanilla已有基本混合SQL、raw缓存与线程执行器；所谓未优化是相对后续优化而言。')
# 03
s=slide('优化沿着关键路径逐层推进',sub='先减少工作量，再增加并行度，最后压低复用与资源管理开销。')
flow(s,[('vanilla','建立按需 SQL 路径'),('v1 / 第一轮','查询修复 · 去重 · 缓存'),('v1 / 第二轮','encoded · direct · 线程'),('v2','多进程 · 共享 · 磁盘')],2.65,w=2.59,h=1.4)
box(s,.73,4.55,11.87,1.06,NAVY,radius=True);txt(s,1,4.79,11.2,.57,'v2 后续：HM 传输粒度  →  Salt/Stack 共享查询  →  mmap 与 WAL 长跑治理',19,WHITE,True)
takeaway(s,'历史文档中的 v1 是当时版本，不等于今天 relconnector/ 的全部代码。')
note(s,'早期v1被复制为cache_baseline，随后接受正确性修订，再继续动态特征优化。v2具有独立命名空间。源码中未提交的修改也属于当前工作区，因此不能将这些状态称为固定发布tag。')
# 04
s=slide('GPU 等待特征，成为最明确的优化信号',sub='历史瓶颈定位：ready queue 等待占训练时间的比例',source='来源：doc/bottleneck-analysis-and-optimization.md｜历史版本，仅用于解释选题依据')
for i,(name,v) in enumerate([('Amazon',97.3),('Arxiv',89.0),('Stack',86.5)]):
 y=2.55+i*.85;txt(s,.9,y,1.5,.5,name,20,INK,True);box(s,2.65,y+.07,7.8,.32,LIGHT);box(s,2.65,y+.07,7.8*v/100,.32,TEAL);txt(s,10.8,y-.02,1.5,.5,f'{v}%',23,TEAL,True)
txt(s,.9,5.36,11.8,.6,'特征生产：SQL → 解码 → 文本编码 → TensorFrame → GPU',21,INK,True)
takeaway(s,'阶段服务时间会重叠；判断瓶颈应结合 wall time、ready 等待和具体操作计时。')
note(s,'这些比例来自历史报告，不代表当前v2。异步worker累计操作时间有并行与嵌套，不能相加得到总时长。重点解释为何先优化CPU供给而不是GNN kernel。','bottleneck-analysis-and-optimization.md')
# 05
s=slide('第一轮优化：让 SQL 真正只取需要的行',sub='保留零基 node ID 契约，在数据库查询边界使用可索引的 rowid。')
card(s,.73,2.5,5.72,2.5,'原始问题','WHERE / ORDER BY 使用 rowid − 1\n表达式可能破坏直接 rowid 查找\n少量采样请求触发大范围扫描',ORANGE)
card(s,6.7,2.5,5.9,2.5,'修复路径','WHERE rowid IN (...)\nORDER BY rowid\n仅在 SELECT 输出中转换为零基 ID')
txt(s,.88,5.36,11.5,.53,'进一步：稀疏请求合并 IN 查询，稠密访问按块读取并复用 raw cache。',19)
takeaway(s,'当前 vanilla 已回补 rowid 修复，旧版超时不能直接代表今天的 vanilla。')
note(s,'无显式主键表对外暴露rowid-1，查询输入使用node ID+1。后续离线工具建立显式连续node ID。该修复后来回补到vanilla，正式对照应重跑修订后的版本。','feature-pipeline-optimization-plan.md；cache-baseline-correctness-report.md')
# 06
s=slide('去重的是特征生产，不是训练样本',sub='批内、推荐分支及跨 batch window 合并静态特征请求，再恢复原顺序。')
flow(s,[('请求节点','[7, 3, 7, 8]'),('唯一节点','[3, 7, 8]'),('读取与编码','每个唯一节点处理一次'),('inverse 恢复','[f7, f3, f7, f8]')],2.7,w=2.59)
card(s,.73,4.43,5.73,1.45,'扩大复用范围','source / positive / negative 及相邻 batch',TEAL)
card(s,6.7,4.43,5.9,1.45,'保持语义边界','保留重复位置、label、边与 seed time',BLUE)
takeaway(s,'窗口受 batch / byte 上限约束；时间相关表示仍按各自 batch 计算。')
note(s,'早期采用4 batch、512 MiB feature window。扩大窗口可以降低重复工作，也会增加首批延迟和临时内存。跨epoch预取允许提前准备下一轮，但fetch窗口本身仍需按epoch拆分。','feature-pipeline-optimization-plan.md')
# 07
s=slide('第一轮收益取决于数据的重复结构',sub='历史 vanilla 与早期 v1：同轮次 wall time，越短越好。',source='来源：doc/gpu-feature-optimization-experiment.md｜正确性修复前历史证据')
txt(s,.95,2.18,6,.3,'灰：历史 vanilla     绿：早期 v1',12,MUTED)
paired(s,[('Salt',14132.8,2832.6),('Stack',7831.8,6428.7)],15000,y=2.85,w=5.6)
card(s,10,2.65,2.59,2.85,'4.99× / 1.22×','Salt：高节点复用\nStack：低节点复用\n整句 LRU 仍抖动',TEAL)
txt(s,.95,5.4,11.2,.52,'重复因子：Salt 2.114；Stack 1.011。相同优化在不同工作集上的收益不同。',17)
takeaway(s,'这些历史数字解释优化方向，不是当前三个版本的配对实验成绩。')
note(s,'Salt raw覆盖率98.70%，Stack仅4.86%。Stack文本缓存发生约4390万次淘汰，工作集超过缓存容量。历史v1九项完成，Ratebeer四小时只完成15686/20743。','gpu-feature-optimization-experiment.md')
# 08
s=slide('正确性修订为后续性能比较建立共同契约',sub='优化执行过程，保持官方特征类型、任务隐藏规则和采样语义。')
for x,label,body,c in [( .73,'输入一致','共同读取官方 stype artifact\n按任务语义隐藏目标列\n不以性能为由删列或改型',TEAL),(4.77,'结构一致','保留节点 / 边顺序\n保留重复项与缺失位置\n只复用静态特征',BLUE),(8.81,'数值验证','固定输入比较 TensorFrame\n比较 loss 与参数更新\nGPU 与调度变化单独分析',ORANGE)]:card(s,x,2.65,3.79,2.93,label,body,c)
takeaway(s,'修订标识：relbench-correctness-v1；修复前后的 loss / 性能记录不能混用。')
note(s,'已记录固定fixture四版本逐值一致及真实F1推荐60组特征一致，但不能外推所有任务和调度都严格等价。current vanilla回补了SQL问题，历史实现必须使用归档。ready顺序会改变优化器轨迹。','cache-baseline-correctness-report.md')
# 09
s=slide('第二轮 v1：把复用提升到最终静态特征',sub='文本向量命中仍需组装；encoded cache 进一步省掉反复的 TensorFrame 构造。')
rows=[('RAW CACHE','解码后的 DataFrame 块','省 SQL / 解码',BLUE),('TEXT CACHE','冻结的 GloVe 向量','省文本模型执行',TEAL),('ENCODED CACHE','最终静态 CPU TensorFrame','省 prepare / converter',ORANGE)]
for i,(a,b,c,co) in enumerate(rows):
 y=2.48+i*.92;box(s,.73,y,11.87,.72,WHITE,radius=True);txt(s,.95,y+.18,2.9,.4,a,18,co,True);txt(s,3.9,y+.18,4.4,.4,b,18);txt(s,8.7,y+.18,3.6,.4,c,17,co,True)
txt(s,.94,5.57,11.4,.38,'按 (table, node_id) 找到特征，仅编码 miss，再按采样顺序恢复。',18)
takeaway(s,'只缓存冻结 / 静态表示，不缓存会随训练参数更新的 GNN 隐状态。')
note(s,'encoded cache整块持有TensorFrame，按tensor storage、ID和索引成本计费。有界LRU、oversized rejection及二次访问准入帮助保护热点维表。','adaptive-feature-optimization-report.md；../relconnector/features/cache.py')
# 10
s=slide('降低执行开销，但不默认“并行越多越快”',sub='direct GloVe、连续 slab、线程流水线、自适应窗口与初始化缓存。')
card(s,.73,2.55,5.73,2.66,'机制','direct：复用官方 tokenize / lookup / pool\nslab：减少小 Tensor 与 storage 对象\n线程：SQL 与 encode 重叠\n窗口：按重复率与缓存覆盖率调整')
card(s,6.7,2.55,5.9,2.66,'已有短程证据','文本微基准：局部约 1.59×\nSalt GPU 2×50：训练耗时 ↓34.0%\nRatebeer GPU 2×20：↓8.5%\nF1 CPU 小任务：反而退化约 1.5%',BLUE)
takeaway(s,'局部快路径和短程组合消融不能直接外推完整 epoch 的端到端收益。')
note(s,'direct与official固定文本逐值相同。Salt/Ratebeer实验同时改变direct、worker与encoded缓存，无法拆分单项贡献。初始化缓存持久化schema统计量和纯拓扑，冷暖差异必须分开计费。','adaptive-feature-optimization-report.md')
# 11
s=slide('v2：围绕特征生产重设进程边界',sub='v1 单进程优化后，assemble 成为瓶颈；v2 将采样、组装、训练分进程执行。',source='来源：relconnector_v2/runtime/process_pipeline.py 与 process_api.py')
flow(s,[('采样进程','seed 流 + 共享拓扑'),('有界 plan','窗口与 inflight 限制'),('组装进程池','SQL + GloVe + Frame'),('训练主进程','ready → GPU 更新')],2.65,w=2.59,h=1.45)
box(s,.73,4.6,11.87,1.13,NAVY,radius=True);txt(s,1,4.84,11.2,.6,'worker 内：SQLite reader / raw cache / 字符串 / DataFrame\n跨进程：采样计划与准备好的 tensor',18,WHITE)
takeaway(s,'减少 Python 数据对象跨进程传输，同时为每个 worker 的私有工作集计费。')
note(s,'v2独立命名空间，不导入原训练实现。拓扑tensor共享，prepared tensors通过PyTorch multiprocessing传输。Assembly负责从读取到组装，而不是机械复制v1线程边界。多进程增加模型、连接和临时内存，必须全进程树测量。')
# 12
s=slide('调度要同时控制顺序、供给与资源',sub='有界队列解决积压；strict / ready 控制训练顺序；许可调整活跃 worker。')
card(s,.73,2.52,3.79,3.13,'顺序','strict：按序消费\n慢窗口可能阻塞后续\n\nready：轮内先就绪先训练\nepoch gate 防止跨轮越界',BLUE)
card(s,4.77,2.52,3.79,3.13,'供给','ready 饥饿 → 增加活跃数\n高水位 → 减少活跃数\n\n跨 epoch 可提前准备\n单个窗口仍按 epoch 拆开',TEAL)
card(s,8.81,2.52,3.79,3.13,'预算','各队列按字节设定限额\ninflight 限制未完成窗口\n\n预先 spawn 不等于按需分配\n暂停 worker 不等于释放内存',ORANGE)
takeaway(s,'正式确定性对照先用 strict；ready 的吞吐与训练轨迹影响另行报告。')
note(s,'当前默认ready，矩阵初始活跃1个并可扩展4个。队列高水位为75%。epoch预取不意味着可把不同epoch合入fetch_many。消费者等待数据和临时对象还需纳入内存预算。')
# 13
s=slide('v2 的复用范围：从 worker 内扩展到跨运行',sub='每层缓存解决不同的重复工作，也引入不同的管理成本。')
for i,(a,b,c) in enumerate([('worker 本地','raw / 可选 encoded / 共享槽位索引','快速访问，需控制每进程预算'),('运行期共享','文本向量 + reservation / encoded','避免跨 worker 重复编码'),('磁盘持久化','SQLite 索引 + float32 向量分片','复用超过 RAM 的文本工作集')]):
 y=2.47+i*.98;box(s,.73,y,11.87,.8,WHITE,radius=True);txt(s,.94,y+.22,2.36,.4,a,19,TEAL,True);txt(s,3.4,y+.22,5.4,.4,b,17);txt(s,8.9,y+.22,3.48,.4,c,14,MUTED)
takeaway(s,'磁盘缓存按访问惰性生成；不要求全表特征入 RAM，但新增长期磁盘成本。')
note(s,'shared reservation协调同一miss由一个worker编码。已发布向量槽不可变，支持本地只存位置索引。磁盘缓存不是提前全表物化，也不是零存储成本；访问广泛后可能覆盖大量文本。旧文档不持久化embedding的限制已发生演进。')
# 14
s=slide('磁盘缓存：模型调用归零，不等于总耗时归零',sub='Arxiv 2×50 batch｜磁盘缓存 cold / warm / off 的已有短测',source='来源：doc/v2-disk-text-cache-experiment-20260925.md｜两组初始化/文本缓存状态须区分')
for i,(n,v,co,detail) in enumerate([('cold',396.6,ORANGE,'首次初始化与写入'),('off',204.7,'AAB9C8','574,119 次模型输入'),('warm',190.5,TEAL,'模型输入为 0')]):
 y=2.55+i*.83;txt(s,.95,y,1.3,.4,n,20,INK,True);box(s,2.35,y+.05,6.25*v/400,.33,co);txt(s,2.49+6.25*v/400,y-.02,1.6,.4,f'{v:.1f} s',18,co,True);txt(s,10.12,y+.02,2.38,.65,detail,14,MUTED)
txt(s,.95,5.38,11.6,.52,'warm 相比 off 仅快约 6.9%：frame conversion、缓存查询与展开仍然存在。',19,INK,True)
takeaway(s,'cold / warm 的 2.08× 还含首次初始化成本，不能全部归因于避免文本编码。')
note(s,'adaptive磁盘激活阈值：DB>=256MiB、模型输入>=4096、模型时间>=1秒、内存命中<=10%。F1保持内存路径，磁盘未激活。这里两个worker，ready顺序；不做数值逐位等价结论。','v2-disk-text-cache-experiment-20260925.md')
# 15
s=slide('HM：高命中缓存，被整块 RPC 传输拖慢',sub='减少传输粒度，再评估 worker 本地 encoded cache。',source='来源：doc/v2-hm-regression-20260925.md｜完整两轮 14,972 batch')
card(s,.73,2.55,5.3,2.87,'根因 → 修复','article 命中率 99.77%\n只请求少量行，却返回整块 TensorFrame\n\n共享缓存仅传命中行\n提供 local encoded，避免反复 RPC')
txt(s,6.6,2.58,5.7,.4,'两轮 pipeline 时间',18,MUTED)
txt(s,6.6,3.18,5.7,.8,'4,603 → 1,756 s',31,TEAL,True)
txt(s,6.6,4.3,5.7,.8,'2.62×\nready 等待：3,349 → 482 s',23,INK,True)
takeaway(s,'最终配置由 shared 2 GiB 改为 local 总计 4 GiB；并非单一因素或等容量完整对照。')
note(s,'短程等预算对照支持通信归因：shared206.0s、只传行183.5s、local2GiB157.7s、local4GiB152.8s。完整端到端2.87x包含冷暖初始化差异，pipeline2.62x仍包含作用域/容量改变。矩阵HM仍显式特配local4GiB。','v2-hm-regression-20260925.md')
# 16
s=slide('Salt：99.93% 命中，仍被共享查询拖慢',sub='向量保持共享；本地只存已发布槽位索引，让重复访问绕过共享锁。',source='来源：doc/v2-salt-regression-20260926.md｜完整两轮 6,340 batch')
flow(s,[('观察','查询成本 + 实际复用'),('高收益时启用','文本 → 共享槽位'),('重复命中','无共享锁批量 gather'),('低收益时退出','释放索引 + 冷却')],2.5,w=2.59,h=1.23)
txt(s,.94,4.2,7.7,.8,'18,260 → 5,208 s',33,TEAL,True);txt(s,9.05,4.24,3.4,.8,'3.51×',37,TEAL,True)
txt(s,.95,5.24,11.5,.5,'完整修复前后模型输入保持 146,635 / 0；没有靠减少 batch 获得提升。',18)
takeaway(s,'3.51× 是修复旧 v2 的收益；相对历史 v1 的单位 batch 观测仅约 1.088×。')
note(s,'本地位置索引每worker估算32MiB，向量不重复复制，reservation不变。F1未激活；Arxiv低复用激活后回退。最终完整Salt还包含共享查询批量化，不能视为单独位置索引的因果倍数。ready等待仍约64.7%。','v2-salt-regression-20260926.md')
# 17
s=slide('Stack：从共享查询开销，到长跑资源耗尽',sub='性能修复与可靠性修复，需要沿同一生产链路验证。',source='来源：doc/v2-stack-regression-20260926.md｜暖磁盘缓存条件')
card(s,.73,2.53,5.73,2.85,'共享查询批量化','哈希移到锁外；元数据零拷贝视图\n锁内取槽地址；锁外批量 gather\n批量发布，限制取锁等待\n\n100 batch：296.04 → 53.45 s')
card(s,6.7,2.53,5.9,2.85,'长跑资源有界','每 cache 最多 1,024 个 mmap\nLRU 关闭映射，不删除向量\n同步错误通道避免报错时新建线程\n\n完整 13,228 batch：3,665.66 s',BLUE)
takeaway(s,'完整运行跨过旧挂起点并正常退出；早期无调用栈的停滞不能统一断言为死锁。')
note(s,'短测后测继承少量向量并受同机负载影响，5.54x不是严格单因素。后续长跑已有77339个被索引分片，资源逼近vm.max_map_count；隔离复现mmap分配失败及线程无法启动。修复后映射保持1024，完整训练模型输入0/6942。','v2-stack-regression-20260926.md')
# 18
s=slide('Amazon：资源回收也需要匹配磁盘工作负载',sub='WAL 无节制增长与过于频繁 checkpoint，都可能拖慢训练。',source='来源：doc/v2-amazon-regression-20260926.md｜仅短程成功结果')
flow(s,[('旧问题','WAL 约 60.46 GB'),('固定小阈值','64 MiB 频繁刷盘'),('自适应回收','按索引大小 / 空间调整'),('分片追加','每 writer 8,192 行/片')],2.58,w=2.59,h=1.38)
card(s,.73,4.45,5.73,1.46,'短测完成','2×100 batch：676.01 s',TEAL)
card(s,6.7,4.45,5.9,1.46,'资源回收','WAL 峰值约 1.35 GiB，退出归零',BLUE)
takeaway(s,'这是资源回收短测；完整矩阵结果待补，续跑已使用部分热缓存。')
note(s,'默认自适应checkpoint阈值通常64MiB–4GiB，参考索引4倍并受剩余空间1/16约束。触发值不是绝对硬上限。固定64MiB在机械盘上导致零散脏页刷出和写锁等待，候选弃用。续跑保留/合并6584936条向量。','v2-amazon-regression-20260926.md')
# 19
s=slide('当前结果：部分任务显著改善，仍有回归与缺口',sub='历史 v1 / v2 的平均单位 batch 耗时比；>1 表示本组记录中 v2 更快。',source='来源：报告 §8 / v2.jsonl｜8 个任务最新成功记录；不是统一最终源码重跑')
ratios=[('Arxiv',5.817),('Stack',3.508),('HM',1.981),('Avito',1.490),('F1',1.197),('Salt',1.088),('Event',.881),('Trial',.792)]
x=2.1;scale=1.18;y0=2.33
box(s,x+scale,y0-.08,.018,3.76,ORANGE);txt(s,x+scale-.2,2.03,1,.23,'1×',11,ORANGE,True)
for i,(name,v) in enumerate(ratios):
 y=y0+i*.455;co=TEAL if v>=1 else RED;txt(s,.91,y,1.15,.32,name,15,INK,True);box(s,x,y+.045,scale*v,.22,co);txt(s,x+scale*v+.12,y-.01,.93,.36,f'{v:.3f}×',14,co,True)
card(s,10.05,2.4,2.55,3.6,'比较限制','历史 A30 / v2 4090\n1 epoch / 2 epochs\n正确性与缓存差异\n\nAmazon /\nRatebeer\n完整结果待补',ORANGE)
takeaway(s,'这是历史趋势图，不是正式配对加速结论；不汇总“十任务总体加速”。')
note(s,'历史v1是早期缓存版，不是当前v1。v2各任务不同时间修复完成，HM local4GiB，其余通常shared2GiB；Arxiv/Stack暖磁盘。报告快照8任务10条成功记录，HM/Salt各保留旧结果。不可计算batch对照。','optimization-evolution-report-20260926.md §8')
# 20
s=slide('有限内存验收：必须统计整个训练系统',sub='v2 的 peak_rss_mb 当前使用 RUSAGE_SELF，只记录训练主进程。')
box(s,.73,2.43,11.87,.8,NAVY,radius=True);txt(s,1,2.61,11.25,.43,'主进程约 2–4 GiB  ≠  整个 v2 训练系统只需 2–4 GiB',25,WHITE,True)
card(s,.73,3.6,3.79,2.27,'额外内存','采样 / Assembly / manager\nGloVe 与临时 Frame\n共享索引、传输与在途数据')
card(s,4.77,3.6,3.79,2.27,'预算边界','ready 默认上限 16 GiB\n队列容量并非实际峰值\n暂停进程不释放全部内存',ORANGE)
card(s,8.81,3.6,3.79,2.27,'完整测量','进程树 PSS / USS\n共享内存、cgroup 峰值\n显存、磁盘、FD / mmap',BLUE)
takeaway(s,'100 万条 300 维 float32 向量仅 payload 约 1.12 GiB，磁盘索引与 WAL 另计。')
note(s,'RUSAGE_SELF源码位于relconnector_v2/benchmark.py。相加RSS会重复计共享页，需PSS/USS和cgroup。默认raw4GiB、encoded2GiB、text512MiB、plan2GiB、ready16GiB，raw/local预算按worker分摊但模型和临时内存额外。不能据此声称32GiB硬限制已验收。')
# 21
s=slide('下一轮实验：把机制证据变成严格系统结论',sub='相同训练语义、相同硬件、相同总内存限制，主对照只保留三个在线版本。')
for x,label,body,co in [( .73,'01  固定边界','冻结 vanilla / v1 / v2\n固定类型、隐藏列与数据版本\nstrict 先测，再测 ready\n完整记录样本与 batch 数',BLUE),(4.77,'02  区分冷热','全冷启动\n初始化暖 / 文本冷\n初始化暖 / 文本热\n记录缓存初始与最终体积',TEAL),(8.81,'03  做关键消融','1 / 2 / 4 worker：等总预算\nencoded：off / shared / local\n文本：shared / adaptive\n磁盘：off / cold / warm',ORANGE)]:card(s,x,2.55,3.79,3.15,label,body,co)
takeaway(s,'先补 Amazon / Ratebeer，诊断 Event / Trial，再统一最终代码重跑已完成任务。')
note(s,'报告中Ratebeer完整历史每轮20743batch，而矩阵max10000，即使达到上限也只是截断工作量。需多次重复、波动与进程树资源，不能将短测或约2%单次差值外推。HM encoded特配应披露或改为可验证运行时策略。')
# 22
s=slide('核心贡献：让有限内存下的特征供给更高效',sub='优化主线：减少重复工作 → 提高生产并行度 → 降低复用成本',dark=True)
for i,(a,b) in enumerate([('按需','从全量特征驻留，转为按采样读取'),('复用','从原始行与文本，提升到最终静态特征'),('并行','用有界多进程让特征生产与 GPU 更新重叠'),('长跑','同时治理共享查询、mmap 与 WAL 成本')]):
 y=2.52+i*.77;txt(s,.83,y,1.35,.5,a,25,TEAL,True);txt(s,2.45,y+.05,9.7,.5,b,21,WHITE)
txt(s,.85,6.08,11.7,.53,'现有机制与定向回归已有证据；统一版本的性能与总内存结论仍需配对验收。',16,'BCD0DF')
note(s,'收束时不强调一个全面加速倍数。项目最有说服力的贡献是明确瓶颈、设计符合内存限制的复用/并行机制，以及对多进程长跑退化的诊断修复。后续用严格实验确认普适性。')
# 23
s=slide('附录 A｜结果数值与工作量',sub='报告快照中每任务最后一条成功记录；v2 两轮，历史 v1 一轮。',source='来源：optimization-evolution-report-20260926.md §8；v2-full-matrix-20260925/v2.jsonl')
rows=[('Arxiv','2,088','981.64','0.47013','5.817×'),('Avito','8,644','1,604.83','0.18566','1.490×'),('Event','76','2,193.96','28.86783','0.881×'),('F1','12','26.57','2.21388','1.197×'),('HM','14,972','1,760.13','0.11756','1.981×'),('Salt','6,340','5,208.23','0.82149','1.088×'),('Stack','13,228','3,665.66','0.27711','3.508×'),('Trial','918','656.95','0.71563','0.792×')]
table(s,['任务','v2 batches','v2 wall (s)','v2 s/batch','历史 v1 / v2'],rows,[1.7,2.2,2.7,2.5,2.77],y=2.25,rowh=.40,size=14)
txt(s,.9,6.12,11.6,.57,'Amazon、Ratebeer 未纳入。硬件、缓存、正确性版本与训练顺序不同，详见第 19 页。',15,ORANGE,True)
note(s,'历史v1 wall分别Arxiv2855、Avito1195.8、Event966.1、F115.9、HM1743.1、Salt2832.6、Stack6428.7、Trial260.2秒。比值按各自wall/batches计算。JSONL的wall保留全精度后取三位比值；表只展示两位wall。报告链接中有完整来源。')
# 24
s=slide('附录 B｜来源、复现与讲解备注',sub='演示依据已完成的详细报告；每页备注记录来源与重要实验条件。')
card(s,.73,2.5,5.73,3.23,'文档与实验依据','optimization-evolution-report-20260926.md\ngpu-feature-optimization-experiment.md\nadaptive-feature-optimization-report.md\nv2-{hm,salt,stack,amazon}-regression-*\nv2-disk-text-cache-experiment-20260925.md',BLUE)
card(s,6.7,2.5,5.9,3.23,'源码与产物','relconnector_v2/runtime/process_pipeline.py\nfeatures/shared_text.py / shared_encoded.py\nfeatures/adaptive_text_index.py / disk_text.py\nbenchmarks/v2-full-matrix-20260925/v2.jsonl\n本目录 build_slides.py 可重新生成 PPTX',TEAL)
takeaway(s,'本演示未重新运行训练；详细报告与历史记录中的测试成绩均保留原始口径。')
note(s,'完整路径均相对项目relconnector/。源码实际位于relconnector_v2/relconnector_v2/。工作区快照包括未提交修改，不对应固定发布tag。PPT以原生文本框、矩形和条形图形构建，均可编辑；附PDF便于预览分享。')

assert len(slides)==24
R.core_properties.title='RelConnector：有限内存下的关系图训练优化演进'
R.core_properties.subject='vanilla → v1 → v2；技术汇报'
R.core_properties.author='RelConnector'
R.core_properties.keywords='在线训练,有限内存,多进程,特征缓存,RelConnector'
# Basic geometry validation before exporting.
for i,s in enumerate(slides,1):
 for sh in s.shapes:
  assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=R.slide_width+20 and sh.top+sh.height<=R.slide_height+20,(i,sh.name)
R.save(OUT/'RelConnector-优化演进.pptx')
(OUT/'speaker-notes.md').write_text('# RelConnector 演讲备注\n\n'+'\n\n'.join(f'## {i:02} · {title}\n\n{body}' for i,title,body in notes),encoding='utf-8')
(OUT/'README.md').write_text('''# RelConnector 优化演进幻灯片

- `RelConnector-优化演进.pptx`：24 页、16:9，可编辑文本与图形，含逐页讲解备注。
- `RelConnector-优化演进.pdf`：便于查看与分享的演示副本。
- `speaker-notes.md`：讲解备注与来源。
- `build_slides.py`：生成源码，依赖 `python-pptx`。

建议讲解时长 20–30 分钟。数据固定于详细报告的 2026-09-26 快照，不代表实时矩阵状态。batch 仅作传统训练背景。

生成：`python build_slides.py`。PDF 使用 LibreOffice 从 PPTX 导出。
字体：Noto Sans CJK SC；跨平台缺少此字体时可安装 Noto CJK，或使用 PDF 保持版式。
''',encoding='utf-8')
print('Saved',OUT/'RelConnector-优化演进.pptx',len(slides),'slides')
