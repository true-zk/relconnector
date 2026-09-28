from pathlib import Path
from copy import deepcopy
import csv,json,statistics as stats
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];SRC=ROOT/'doc/slides/bottleneck-review-20260927'
d=json.loads((SRC/'performance-data.json').read_text());r=list(csv.DictReader((SRC/'rss-data.csv').open()))
vr=100*(1-stats.mean(float(x['v2_main_process_rss_gib'])/float(x['vanilla_reported_rss_gib']) for x in r))
vt=100*(1-stats.mean(x['v2_s_per_batch']/x['vanilla_s_per_batch'] for x in d if x['vanilla_s_per_batch'] is not None))
br=stats.mean(float(x['batch_reported_rss_gib'])/float(x['v2_main_process_rss_gib']) for x in r)
bt=stats.mean(x['v2_s_per_batch']/x['batch_s_per_batch'] for x in d)
metrics={'aggregation':'equal-weight arithmetic mean of per-dataset relative ratios','rss_dataset_count':10,'vanilla_time_datasets':[x['dataset'] for x in d if x['vanilla_s_per_batch'] is not None],'batch_time_dataset_count':10,'vanilla_rss_record_reduction_pct':vr,'vanilla_time_reduction_pct':vt,'batch_rss_record_over_v2_main_rss_ratio':br,'v2_over_batch_time_ratio':bt,'v2_over_batch_time_increase_pct':(bt-1)*100,'warning':'RSS records are not comparable total-process-tree memory: v2 only RUSAGE_SELF. Time is end-to-end seconds/batch, includes preparation and mixed cache states; Vanilla time excludes five timeout datasets.'}
(OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
P=Presentation();P.slide_width=Inches(13.333);P.slide_height=Inches(7.5)
s=P.slides.add_slide(P.slide_layouts[6])
def text(x,y,w,h,t,size=18,color='263746',bold=False):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if not i else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK SC';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(3)
 return sh

def box(x,y,w,h,color):
 sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(color);sh.line.fill.background()
text(.6,.25,12,.25,'08 / 汇总结论 · 历史实测记录',11,'657789',True)
text(.6,.72,12,.56,'v2：相对 Vanilla 提速，相对 Batch 权衡',27,bold=True)
text(.62,1.35,12,.35,'逐数据集计算相对变化，再等权取算术平均；耗时统一为端到端秒 / batch。',13,'657789')
box(.6,1.96,12.13,1.56,'EEF4FA')
text(.85,2.12,2.3,.34,'相比 Vanilla',17,'365F85',True)
text(.85,2.61,11.6,.7,f'已记录 RSS 平均降低 {vr:.1f}%；在双方均完成的 5 个数据集上，\n单位 batch 耗时平均降低 {vt:.1f}%。',22,bold=True)
box(.6,3.77,12.13,1.56,'EAF5EF')
text(.85,3.93,2.3,.34,'相比 Batch',17,'39715A',True)
text(.85,4.42,11.6,.7,f'Batch / v2 的 RSS 记录值比平均为 {br:.2f} 倍；\nv2 单位 batch 耗时平均增加 {(bt-1)*100:.1f}%（为 Batch 的 {bt:.2f} 倍）。',22,bold=True)
box(.6,5.6,12.13,.46,'FFF2E1')
text(.78,5.67,11.8,.29,'RSS 口径限制：v2 仅统计主进程，不含采样、组装和 manager；上述比值不代表整体内存降幅。',12,'915E25',True)
text(.65,6.24,12,.63,'RSS 与 Batch 耗时：10 库；Vanilla 耗时：Event、F1、Salt、Stack、Trial，剔除其余 5 项超时。\n历史超时 RSS 为终止前峰值；v2 为两轮、缓存状态混合，Ratebeer 截断；结果不是严格配对实验。',11,'657789')
text(.65,7.12,12,.2,'来源：历史实验报告 / rss-data.csv / performance-data.json；未使用本次内存估算。',9,'81909C')
notes='本页使用历史报告与先前选定 v2 实测记录，未使用内存估算。\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n公式：Vanilla RSS = mean(1-v2_RSS_i/vanilla_RSS_i)；Vanilla 耗时 = mean(1-v2_sec_per_batch_i/vanilla_sec_per_batch_i)，仅5个完成项；Batch RSS 倍数 = mean(batch_RSS_i/v2_main_RSS_i)；Batch 耗时增加 = mean(v2_sec_per_batch_i/batch_sec_per_batch_i-1)。\n平均比值与均值之比不同。本页 Batch 耗时1.753×是数据集等权平均；旧页按实际batch数量折算的合计比值2.300×是另一种加权方式，不应混用。RSS不支持整体内存优化的因果结论。\n来源：doc/gpu-feature-optimization-experiment.md；doc/slides/bottleneck-review-20260927/rss-data.csv；doc/slides/bottleneck-review-20260927/performance-data.json；relconnector_v2/relconnector_v2/benchmark.py:98。'
s.notes_slide.notes_text_frame.text=notes
P.save(OUT/'RelConnector-v2两组对比总结.pptx')
base=ROOT/'doc/slides/framework-architectures-20260927/RelConnector-优化复盘-含四版架构图.pptx'
full=Presentation(base);dst=full.slides.add_slide(full.slide_layouts[6])
for sh in s.shapes:dst.shapes._spTree.insert_element_before(deepcopy(sh.element),'p:extLst')
dst.notes_slide.notes_text_frame.text=notes
# Place summary after the 19-page report and before the four architecture appendices.
ids=full.slides._sldIdLst;last=ids[-1];ids.remove(last);ids.insert(19,last)
full.save(OUT/'RelConnector-优化复盘-含架构与对比总结.pptx')
assert len(full.slides)==24
(OUT/'README.md').write_text('# v2 两组对比总结\n\n新增第20页，四张架构图顺延至第21–24页。\n\n'+notes+'\n')
print(json.dumps(metrics,ensure_ascii=False,indent=2))
