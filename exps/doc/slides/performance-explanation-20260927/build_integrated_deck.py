"""Integrate mechanisms and current evidence into the original narrative, without appendices of updates."""
from pathlib import Path
import json,subprocess,sys
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
TARGET=D/'RelConnector-优化复盘-按数据类型解释.pptx'
MARK='INTEGRATED_V2_20260927'

def update():
 global R
 subprocess.run([sys.executable,str(D/'build_integrated_figures.py')],check=True)
 p=Presentation(TARGET)
 # Remove old appended targeted pages; this makes both the 28- and 30-page source decks valid.
 for i in reversed(range(len(p.slides))):
  s=p.slides[i]
  if s.has_notes_slide and 'TARGETED_BATCH_V2_20260927' in s.notes_slide.notes_text_frame.text:
   sid=p.slides._sldIdLst[i];p.part.drop_rel(sid.rId);p.slides._sldIdLst.remove(sid)
 assert len(p.slides)==28,'Run build_update.py first to rebuild the source narrative.'
 original=list(p.slides._sldIdLst)
 src=(ROOT/'doc/slides/bottleneck-review-20260927/build_slides.py').read_text()
 helpers=src[src.index('R=Presentation()'):src.index('# 1\n')]
 globals().update({'FONT':'Noto Sans CJK SC','INK':'172B3A','BLUE':'3567A8','TEAL':'008879','GREY':'88929C','LIGHT':'EDF2F5','WHITE':'FFFFFF','MUTED':'586B79','ORANGE':'B36A24'})
 exec(helpers,globals());R=p
 refs=['doc/targeted-batch-results-20260927.json','doc/v2-ratebeer-cache-placement-20260927.md','doc/v2-stack-compaction-20260927.md']
 new={}
 def done(key,s,notes):
  note(s,notes,refs);new[key]=p.slides._sldIdLst[-1]
 s=slide('04 / 阶段二：采样需求','采样输出计划，组装只处理唯一物理行','保持时序邻域与逻辑节点实例；合并的是静态特征请求。')
 for x,label,sub in [( .65,'采样计划','节点 / 边 / 时间'),(3.75,'窗口去重','唯一行 + 逆映射'),(6.85,'读取与编码','只生产必要特征'),(9.95,'恢复批次','逻辑位置保持不变')]:node(s,x,2.45,2.7,1.35,label,sub)
 for x in [3.38,6.48,9.58]:line(s,x,3.08,x+.32,3.08,TEAL,True)
 table(s,['访问特征','Salt','Ratebeer'],[['历史请求 / 唯一行','2.114','1.105'],['理论可消除重复行工作','52.7%','9.5%']],[5.8,3.2,3.2],y=4.35,rowh=.48,fontsize=18)
 takeaway(s,'v1 建立窗口复用；v2 将采样与 assemble 分离，让计划生产与特征生产重叠。')
 done('demand',s,'重复因子来自历史诊断；不是采样内核或端到端加速比。单采样进程，不改变邻居采样算法。')
 s=slide('04 / 编码缓存：统一考虑传输粒度与位置','缓存放在哪里，要看访问成本与额外复用','HM 的高复用与 Ratebeer 的低复用，都可能被跨进程访问成本限制。')
 table(s,['机制 / 配置','HM：历史同预算诊断','Ratebeer：本次同预算诊断'],[['共同预算','总计 2 GiB','总计 2 GiB，4 worker'],['共享命中路径','整块 → 只传请求行','查询 + 发布仍昂贵'],['本地布局','减少 RPC 与重建','每 worker 512 MiB'],['观测结果','206.0 → 157.7 秒 / 1000批','1104.6 → 980.2 ms/batch'],['主要取舍','跨进程复用与传输量','命中率 16.5% → 1.6%']],[2.6,4.45,5.15],y=2.15,rowh=.57,fontsize=16)
 text(s,.77,5.86,11.8,.35,'判据：共享多省下的编码成本，能否覆盖多付出的查询、发布与通信成本？',18,True,c=BLUE)
 takeaway(s,'Ratebeer 流水线平均每批减少 11.26%；这是显式配置选择，尚非自动部署。')
 done('placement',s,'HM为旧1000批同预算对照；Ratebeer为64批×每配置两次预热后对照，不能横向比较两库秒数。Ratebeer查询22.378→2.175、发布36.642→8.931工作进程秒；主指标是流水线时间/批数。完整矩阵默认尚未切换。')
 s=slide('04 / 持久化缓存：从命中走向廉价访问','向量已缓存，物理碎片仍会拖慢 assemble','把布局整理融入持久化方案：先复用计算结果，再减少命中后的文件管理。')
 node(s,.7,2.28,3.55,1.42,'原布局：79,587 片','映射上限 1,024 / worker')
 node(s,4.92,2.28,3.55,1.42,'离线合并 + 校验','2,891,661 条向量逐字节一致')
 node(s,9.14,2.28,3.48,1.42,'新布局：353 片','有效集合可留在映射缓存')
 line(s,4.3,3,4.85,3,TEAL,True);line(s,8.51,3,9.07,3,TEAL,True)
 table(s,['Stack：同逻辑缓存，200批 / 次','原布局','合并后'],[['映射打开次数 / 次运行','238,288','1,312.5'],['映射淘汰次数 / 次运行','234,192','0'],['流水线平均 ms/batch','428.6','370.6']],[6.1,3.05,3.05],y=4.09,rowh=.48,fontsize=17)
 takeaway(s,'在线每批减少 13.52%；离线整理 144.43 秒另计，不覆盖原缓存。')
 done('layout',s,'原布局A2/A3、合并布局B2/B3，每组2次；文件缓存预热后。磁盘查询工作进程时间118.304→54.904秒，下降53.59%。显式离线工具，按分片数/映射预算筛选，不按数据集名判断。不是运行时自动迁移，不宣称RSS下降。')
 s=slide('04 / 特征复制与读取：以关键路径决定取舍','少做一个操作，不一定让一批更快','同一套验收原则覆盖已保留的优化和已撤回的候选。')
 table(s,['尝试','局部观察','端到端结果','当前处理'],[['CPU 行选择','Salt 展开 −17.03%','91.58 → 88.63 秒','保留；短测 −3.22%'],['融合多块缓存投影','查询 −18.90%','128.32 → 129.30 秒','候选归档'],['编码缓存前移到 SQL 前','Avito SQL 行数 −81.56%','共享路径仅 −0.17%','候选归档'],['同前移 / Ratebeer 本地','进一步省去部分读取','仅再减少 0.87%','恢复原读取路径']],[3.15,3.35,3.2,2.5],y=2.25,rowh=.68,fontsize=15)
 text(s,.78,5.94,11.8,.3,'行选择试验与本轮 Ratebeer / Stack 使用不同窗口和基线，百分比不能相加。',16,c=MUTED)
 takeaway(s,'默认方案依据完整工作窗口的实际时间，不能依据累计 worker 时间单独决定。')
 done('copy',s,'Salt每版2次60批；投影试验受外部负载变化影响，无整体收益证据。对应各专项报告。所有操作计时允许重叠，不可相加成墙钟。')
 s=slide('04 / 工作负载 → 成本 → 策略','把新旧优化放在同一套选择逻辑中','数据集只是代表证据；同一任务可同时具备多种特征。')
 table(s,['观察信号','v2 对应机制','证据 / 实现边界'],[['宽表冷统计重','预算分组 + 精确分位数','Event 冷初始化 3.57×'],['重复文本多、共享查询贵','本地槽位索引 + 共享向量','Salt 完整修复记录'],['编码行命中传输贵','请求行投影 + 共享/本地配置','HM；Ratebeer 本地每批 −11.26%'],['文本工作集大','有条件持久化冻结向量','区分首次创建与热复用'],['持久化小分片过多','离线合并 + 映射预算筛选','Stack 每批 −13.52%，整理另计'],['训练批次少','复用离线产物、减少固定开销','F1；少 worker 仍待对照']],[3.4,4.25,4.55],y=2.13,rowh=.56,fontsize=15)
 takeaway(s,'自动机制、显式配置和未来方案分开描述；不把候选写成已部署能力。')
 done('strategy',s,'全部机制在相应前文页解释。本地缓存位置尚非自动学习；离线整理显式调用。')
 s=slide('05 / 当前 v2 定向结果','新的每 batch 结果：供给时间与等待一起下降','同请求、同预算；每种配置重复两次，分别完成 64 / 200 批。')
 s.shapes.add_picture(str(D/'assets/v2-current-per-batch.png'),Inches(.65),Inches(2.03),width=Inches(11.88),height=Inches(3.72))
 text(s,.75,5.96,11.9,.3,'主指标：流水线总时间 / 完成批数；包含内部启动与排空，不是单批独立延迟。',15,c=MUTED)
 takeaway(s,'Ratebeer −11.26% ｜ Stack −13.52%；Stack 一次性整理成本为 144.43 秒。')
 done('current',s,'数据源为targeted-batch-results-20260927.json。等待是流水线的一部分，整次运行又包含流水线；三行不可相加。Ratebeer墙钟每批下降18.71%，Stack12.04%；不得当完整矩阵或RSS更新。')
 s=slide('05 / 十库历史参考 + 当前定向证据','跨版本概览，与本轮 v2 对照分层展示','上方是历史端到端时间与 RSS；下方是本轮配对流水线结果。')
 s.shapes.add_picture(str(ROOT/'doc/figures/vanilla-v1-v2-20260927/final/vanilla-v1-v2-time-rss.png'),Inches(.75),Inches(1.98),width=Inches(11.8),height=Inches(4.91))
 done('historical',s,'历史Vanilla/v1单轮、v2两轮且缓存冷热不同；Event49.83×包含热初始化，冷初始化仅3.57×。Ratebeer v1四小时超时，v2运行15444.93秒完成2×10000批，不是同预算完整遍历。TO柱为状态符号。图中v2 RSS只含主进程。最新64/200批对照不替换历史端到端柱，不外推十库成绩。')
 s=slide('05 / 内存：保留实测口径','新优化测了吞吐，没有产生新的作业内存结论','Batch、Vanilla、v1 为历史报告；v2 空心点仅为主进程 RSS。')
 s.shapes.add_picture(str(D/'assets/rss-current-scope.png'),Inches(.7),Inches(2.12),width=Inches(11.92))
 text(s,.77,5.88,11.8,.36,'Stack 合并减少文件管理，不减少向量载荷；本地缓存也可能复制跨 worker 热点。',17,c=MUTED)
 takeaway(s,'RSS 未更新测量；不再用主进程记录值推导“整个作业内存降低多少倍”。')
 done('memory',s,'来自历史rss-data.csv；Amazon独立线性量程，其余共用另一个量程。此次无新作业级内存测量，不能用新时间结果补造RSS。')
 selected=[*original[:6],original[11],new['demand'],original[7],new['placement'],original[8],new['layout'],original[10],new['copy'],original[12],new['strategy'],new['current'],new['historical'],new['memory'],original[18],*original[24:28]]
 ids=p.slides._sldIdLst
 for sid in list(ids):
  ids.remove(sid)
  if sid not in selected:p.part.drop_rel(sid.rId)
 for sid in selected:ids.append(sid)
 # Retitle the transition, update conclusion, and assign stable page numbers.
 replacements={'命中缓存以后，访问成本仍可能成为瓶颈':'assemble 成为瓶颈：单进程优化后仍供给不足','增加缓存和 worker，并不必然让训练更快。':'先分离采样、组装与训练，再处理多进程缓存的访问成本。','最新 v2 结果':'历史 v2 结果','性能优先级：Amazon / Ratebeer 的特征供给；Event 的图拓扑构建。':'性能优先级：验证 Ratebeer 长窗口；补 Amazon 诊断；优化 Event 拓扑。'}
 manifest=[]
 for i,s in enumerate(p.slides):
  for sh in s.shapes:
   if sh.has_text_frame:
    for para in sh.text_frame.paragraphs:
     for run in para.runs:
      for a,b in replacements.items():run.text=run.text.replace(a,b)
    if sh.top>Inches(7) and sh.left>Inches(12):sh.text_frame.paragraphs[0].text=f'{i+1:02}'
  if i<20:s.notes_slide.notes_text_frame.text=MARK+'\n'+s.notes_slide.notes_text_frame.text
  title=[sh.text for sh in s.shapes if sh.has_text_frame and sh.text][:3]
  manifest.append({'page':i+1,'title':title,'notes':s.notes_slide.notes_text_frame.text})
 p.save(TARGET)
 (D/'integrated-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 (D/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 (D/'speaker-notes.md').write_text('# 整合版逐页说明\n\n'+'\n\n'.join(f"## {m['page']:02} {' / '.join(m['title'][:2])}\n\n{m['notes']}" for m in manifest))
 (D/'targeted-update-manifest.json').write_text(json.dumps({'mode':'integrated','slide_count':len(p.slides),'cache_placement_page':10,'disk_layout_page':12,'current_results_page':17,'historical_reference_page':18,'source':'doc/targeted-batch-results-20260927.json'},ensure_ascii=False,indent=2)+'\n')
 print('Integrated deck:',len(p.slides),'pages; 20 narrative + 4 architecture appendix.')

if __name__=='__main__':
 subprocess.run([sys.executable,str(D/'build_update.py')],check=True)
