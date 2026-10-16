from pathlib import Path
from copy import deepcopy
import hashlib,json
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_SHAPE_TYPE
from pptx.oxml.xmlchemy import OxmlElement
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
OLD=ROOT/'doc/slides/bottleneck-review-20260927'
# Re-render prior raster charts with the v1 display name; retain original input field names.
source=(OLD/'build_slides.py').read_text().split('R=Presentation();R.slide_width')[0]
source=source.replace('OUT=Path(__file__).resolve().parent',f'OUT=Path({str(OUT)!r})').replace('BASE=OUT.parents[2]',f'BASE=Path({str(ROOT)!r})').replace("'v1'","'v1'")
exec(compile(source,str(OLD/'build_slides.py'),'exec'),{'__file__':str(OLD/'build_slides.py')})
P=Presentation();P.slide_width=Inches(13.333);P.slide_height=Inches(7.5)
INK='263746';MUTED='637486';BLUE='365F85';ORANGE='A95331'
manifest=[]
def text(s,x,y,w,h,t,size=17,color=INK,bold=False):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if not i else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK SC';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(3)
 return sh

def rect(s,x,y,w,h,c):
 sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(c);sh.line.fill.background();sh._element.spPr.append(OxmlElement('a:effectLst'))
 style=sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
 if style is not None:
  ref=style.find('{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef')
  if ref is not None:ref.set('idx','0')
 return sh

def slide(title,sub,notes):
 s=P.slides.add_slide(P.slide_layouts[6]);text(s,.55,.28,12.1,.46,title,25,bold=True)
 if sub:text(s,.58,.92,12.1,.5,sub,13,MUTED)
 s.notes_slide.notes_text_frame.text=notes;manifest.append({'title':title,'notes':notes});return s
base_notes='图：doc/figures/vanilla-v1-v2-20260927/final/vanilla-v1-v2-time-rss.png。Time为端到端秒/batch，包含初始化。版本配色固定，v2仅主进程RSS，空心点不含子进程。超时高柱为符号，不是测量高度。Vanilla5项以及v1 Ratebeer在14400秒超时；v2 Ratebeer运行15444.93秒，2×10000批而非完整2轮。冷热缓存与epoch数不同，不是严格配对实验。按照用户要求，图底部小字移入此备注及后续解释页。'
s=slide('十数据集性能对比｜Vanilla · v1 · v2','',base_notes)
s.shapes.add_picture(str(ROOT/'doc/figures/vanilla-v1-v2-20260927/final/vanilla-v1-v2-time-rss.png'),Inches(.35),Inches(1.15),width=Inches(12.63))
groups=[
('按数据集类型解释（1/2）｜特征处理与数据供给',[
('文本编码重、可跨轮复用','例：Arxiv / Amazon / Stack',
 '瓶颈：长文本与大词汇集合使重复编码昂贵，GPU 等待特征。',
 '策略：直接文本编码＋并行组装；适合时持久化 embedding，跨轮复用。',
 '证据：Arxiv 两轮模型输入 184,724 → 1,316；部分收益来自既有热缓存。'),
('复用率高、缓存命中成本高','例：Salt / HM / Trial',
 '瓶颈：减少编码后，共享查询、锁竞争和跨进程传输成为主要成本。',
 '策略：本地槽位索引、批量取向量、只传请求行；必要时用 worker 本地缓存。',
 '证据：HM ready 等待 3,348.9 → 481.9 秒；本地缓存为显式配置。'),
('读取量大、特征供给仍重','例：Ratebeer / Avito',
 '瓶颈：SQL 读取与组装成本高，未命中的绝对编码量仍可很大。',
 '策略：采样 / 多进程组装 / GPU 训练重叠；缓存路径按实际复用成本选择。',
 '边界：Ratebeer 第二轮仍编码约 2,050 万条；更快不等于消除供给瓶颈。')]),
('按数据集类型解释（2/2）｜初始化与固定开销',[
('数值宽表、初始化占主导','例：Event',
 '瓶颈：列多、逐列扫描与排序重复；少量训练无法摊薄初始化成本。',
 '策略：按行数和内存预算分组扫描，精确 partition 分位数，复用初始化产物。',
 '证据：冷初始化 2,164 → 606 秒（3.57×）；图中 49.83× 包含热初始化收益。'),
('小数据集、固定开销占主导','例：F1',
 '瓶颈：启动、进程协调和队列开销，相对实际训练工作量更突出。',
 '策略：避免不必要的磁盘缓存；当前通用流水线收益有限，少 worker 是后续方向。',
 '证据：图中 v1 / v2 仅 1.13×；不能预期小数据集获得与重文本任务同等收益。')])]
for title,items in groups:
 notes='按数据特征和主导瓶颈归类，不是互斥的数据集标签；同一数据集可同时具备多种特征。机制解释由诊断支持，不表示每项都有独立A/B。来源：doc/v2-dataset-performance-review-20260927.md及各专项报告。\n'+json.dumps(items,ensure_ascii=False)
 sub='归类依据：数据特征 → 主导成本 → 对应策略；数据集只作为代表性证据。'
 s=slide(title,sub,notes)
 for i,(kind,examples,problem,fix,evidence) in enumerate(items):
  y=1.57+i*1.63
  rect(s,.55,y,12.23,1.46,'F3F6F9')
  text(s,.78,y+.12,8.6,.31,kind,19,BLUE,True)
  text(s,9.35,y+.16,3.16,.26,examples,12.5,MUTED)
  text(s,.79,y+.56,11.65,.27,problem,15)
  text(s,.79,y+.91,11.65,.27,fix,15)
  text(s,.79,y+1.23,11.65,.20,evidence,10.5,ORANGE)
 if len(items)==2:
  rect(s,.55,5.14,12.23,1.09,'EAF5EF')
  text(s,.8,5.31,11.7,.31,'优化原则：让策略匹配主导成本，而不是给所有数据集堆同一套缓存。',18,'39715A',True)
  text(s,.8,5.8,11.7,.28,'先减少重复工作，再降低命中成本；并行解决供给，预算控制初始化，固定开销限制小任务收益。',13)
 text(s,.65,6.91,12.0,.3,'类别可以重叠；倍率是跨实验观察值，缓存状态不同；RSS 仅统计 v2 主进程。',11,MUTED)
s=slide('Ratebeer｜为何两个 TO，只有 v2 有完成时间？','图中“完成”同时受吞吐、运行规模和预算影响，需要拆开解释。','来源：doc/gpu-feature-optimization-experiment.md；doc/v2-ratebeer-regression-20260926.md。v1吞吐14400/15686=0.918016；v2 15444.92896/20000=0.772246；比值1.18876，单位batch耗时降低15.8787%。v2绝非同4小时预算内完成完整epoch。磁盘缓存关闭；每轮约2024.5万/2050.3万条文本模型输入，计数非唯一文本。共享存储先完成后发布修复了可导致损坏的竞争；首次故障缺输入，不能确认同源。')
for x,name,a,b,c in [( .6,'Vanilla','4 小时超时','目标：1 × 20,743 批','缺少可靠的完成批数记录'),(4.82,'v1 (single process)','4 小时超时','完成：15,686 / 20,743 批','观测吞吐约 0.918 秒 / batch'),(9.04,'v2 (multi process)','4 小时 17 分完成','完成：2 × 10,000 批（截断）','端到端 0.772 秒 / batch')]:
 rect(s,x,1.64,3.7,1.66,'EEF4FA' if x<9 else 'F9EEE8')
 text(s,x+.17,1.8,3.35,.27,name,16,BLUE,True);text(s,x+.17,2.19,3.35,.34,a,20,INK,True)
 text(s,x+.17,2.68,3.35,.53,b+'\n'+c,12,MUTED)
text(s,.72,3.67,11.9,.40,'性能：观测吞吐约快 1.19×，单位 batch 耗时约少 15.9%。',22,ORANGE,True)
text(s,.74,4.31,11.85,.63,'机制：并行采样、SQL / 特征组装与 GPU 训练，改善供给；但 ready 等待仍达 11,389 秒，\n两轮仍各有约 2,000 万条文本模型输入，说明 SQL 与编码瓶颈尚未消除。',17)
text(s,.74,5.29,11.85,.64,'稳定性：先将 TensorFrame 存储转为共享内存，再发布缓存条目，修复并发读取损坏；\n修复后通过本次 20,000 批长跑。这是正确性证据，不能独立解释 1.19× 的性能收益。',17)
rect(s,.6,6.4,12.13,.61,'FFF2E1')
text(s,.78,6.53,11.78,.37,'结论：v2 在本次配置下更快且成功跑完；尚未证明在同一 4 小时预算完成完整两轮。',16,ORANGE,True)
P.save(OUT/'RelConnector-性能图与分类解释.pptx')
full=Presentation(ROOT/'doc/slides/comparison-summary-20260927/RelConnector-优化复盘-含架构与对比总结.pptx')
# Rename presentation-facing text while preserving run formatting and source-file names in notes.
for oldslide in full.slides:
 for sh in oldslide.shapes:
  frames=[]
  if sh.has_text_frame:frames.append(sh.text_frame)
  if sh.has_table:frames.extend(cell.text_frame for row in sh.table.rows for cell in row.cells)
  for frame in frames:
   for par in frame.paragraphs:
    for run in par.runs:run.text=run.text.replace('v1','v1')
# Replace old chart assets by their exact image hashes.
replacements={hashlib.sha256((OLD/'assets'/f'{n}.png').read_bytes()).hexdigest():(OUT/'assets'/f'{n}.png').read_bytes() for n in ['performance_a','performance_b','rss']}
for oldslide in full.slides:
 for sh in oldslide.shapes:
  if sh.shape_type==MSO_SHAPE_TYPE.PICTURE:
   digest=hashlib.sha256(sh.image.blob).hexdigest()
   if digest in replacements:oldslide.part.related_part(sh._element.blipFill.blip.rEmbed)._blob=replacements[digest]
for k,src in enumerate(P.slides):
 dst=full.slides.add_slide(full.slide_layouts[6])
 # This is required to remap picture relationship IDs when copying the chart.
 for sh in src.shapes:
  if sh.shape_type==MSO_SHAPE_TYPE.PICTURE:
   import io
   dst.shapes.add_picture(io.BytesIO(sh.image.blob),sh.left,sh.top,sh.width,sh.height)
  else:dst.shapes._spTree.insert_element_before(deepcopy(sh.element),'p:extLst')
 dst.notes_slide.notes_text_frame.text=src.notes_slide.notes_text_frame.text
 ids=full.slides._sldIdLst;last=ids[-1];ids.remove(last);ids.insert(20+k,last)
assert len(full.slides)==28
full.save(OUT/'RelConnector-优化复盘-按数据类型解释.pptx')
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('Saved four new pages; integrated deck has 28 slides, new pages 21–24.')

# Include the targeted v2 per-batch evidence after rebuilding the historical deck.
from update_targeted_optimizations import update as update_targeted
update_targeted()
