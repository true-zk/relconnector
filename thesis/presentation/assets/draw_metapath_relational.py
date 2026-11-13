"""Draw a MetaRTL-style relational-table / meta-path explanation figure.

Outputs an editable-text SVG, a high-resolution transparent PNG, and a PDF.
"""
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager
from fontTools.ttLib import TTFont
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, RegularPolygon, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
for weight in ('Regular', 'Bold'):
    source = Path(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc')
    extracted = Path(f'/tmp/codex-NotoSansCJKSC-{weight}.otf')
    if not extracted.exists():
        TTFont(str(source), fontNumber=2).save(str(extracted))
    font_manager.fontManager.addfont(str(extracted))
mpl.rcParams['font.family'] = 'Noto Sans CJK SC'
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42

C = {
    'paper': '#F18C45',      # MetaRTL: Paper
    'author': '#537BC8',     # MetaRTL: Author
    'firm': '#3FC2B7',       # MetaRTL: Firm
    'dark': '#233041',
    'muted': '#596776',
    'border': '#394554',
    'light': '#F5F8FB',
    'blue': '#06438F',
    'orange_path': '#D97019',
    'teal_path': '#148E87',
    'pale_orange': '#FFF4E9',
    'pale_teal': '#E9F8F5',
}

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


def txt(x, y, s, size=15, color=None, weight='normal', ha='left', va='center', **kwargs):
    return ax.text(x, y, s, fontsize=size, color=color or C['dark'], fontweight=weight,
                   ha=ha, va=va, fontfamily='Noto Sans CJK SC', **kwargs)


def panel(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.08,rounding_size=0.20', facecolor='white',
        edgecolor=C['border'], linewidth=1.25, linestyle=(0,(5,4))))
    txt(x+0.24, y+h-0.40, label, 17, C['blue'], 'bold')
    ax.plot([x+0.23,x+w-0.23],[y+h-0.72,y+h-0.72],color='#CBD6DF',lw=1.0)


def table(x, top, w, title, fields, rows, fill):
    row_h=0.42
    h=0.46+0.37+row_h*len(rows)+0.13
    y=top-h
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=C['border'],lw=1.1))
    ax.add_patch(Rectangle((x,top-0.46),w,0.46,facecolor=fill,edgecolor=C['border'],lw=1.0))
    txt(x+w/2, top-0.23, title, 14, 'white', 'bold', 'center')
    ax.add_patch(Rectangle((x,top-0.83),w,0.37,facecolor='#F0F4F8',edgecolor='none'))
    txt(x+w/2,top-0.645,fields,10.8,C['muted'],ha='center')
    for i,row in enumerate(rows):
        ry=top-0.83-row_h*(i+1)
        if i%2==1:
            ax.add_patch(Rectangle((x,ry),w,row_h,facecolor='#FAFCFE',edgecolor='none'))
        ax.plot([x,x+w],[ry,ry],color='#E1E7EC',lw=0.65)
        txt(x+w/2,ry+row_h/2,row,13,C['dark'],ha='center')
    return y,h


def arrow(x1,y1,x2,y2,color='#2F3947',lw=1.5,scale=16):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=scale,
        color=color,lw=lw,shrinkA=0,shrinkB=0))


def node(x,y,kind,label,r=0.28,target=False):
    if kind=='P':
        ax.add_patch(Circle((x,y),r,facecolor=C['paper'],edgecolor='#E43934' if target else C['border'],lw=2.0 if target else 1.25,zorder=6))
    elif kind=='A':
        ax.add_patch(Rectangle((x-r,y-r),r*2,r*2,facecolor=C['author'],edgecolor=C['border'],lw=1.2,zorder=6))
    else:
        ax.add_patch(RegularPolygon((x,y),3,radius=r*1.2,orientation=0,
            facecolor=C['firm'],edgecolor=C['border'],lw=1.2,zorder=6))
    txt(x,y-0.002,label,13,'white','bold','center',zorder=8)


def connector(a,b,color,lw=2.4):
    ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,zorder=2,solid_capstyle='round'))


# Heading and color legend.
txt(0.52,8.56,'元路径如何刻画关系表格中的跨表语义',26,C['blue'],'bold')
txt(0.54,8.07,'示例：从目标论文 P1 出发，寻找其他论文的关联信息',13,C['muted'])
node(11.24,8.47,'P','P',r=0.19)
txt(11.52,8.47,'论文',12,C['dark'])
node(12.63,8.47,'A','A',r=0.19)
txt(12.91,8.47,'作者',12,C['dark'])
node(14.02,8.47,'F','F',r=0.19)
txt(14.30,8.47,'机构',12,C['dark'])
ax.plot([0.52,15.48],[7.89,7.89],color=C['blue'],lw=1.1)

panel(0.48,1.27,4.76,6.40,'① 关系表格：主外键定义连接')
panel(5.48,1.27,5.00,6.40,'② 行记录构成异构关系图')
panel(10.75,1.27,4.76,6.40,'③ 元路径区分复合关系语义')

# Left: three actual tables, with their FK fields visible.
top=6.70
table(0.72,top,1.65,'论文表 P','pid  author_id', ['P1     A1','P2     A1','P3     A2'],C['paper'])
table(2.66,top,1.30,'作者表 A','aid  firm_id',['A1   F1','A2   F1'],C['author'])
table(4.17,top,0.84,'机构 F','fid',['F1'],C['firm'])
arrow(2.40,5.15,2.61,5.15,C['border'],1.2,12)
arrow(3.98,5.15,4.12,5.15,C['border'],1.2,12)
txt(1.92,4.64,'外键',10.5,C['muted'],'normal','center')
txt(4.03,4.64,'外键',10.5,C['muted'],'normal','center')
ax.plot([0.78,4.96],[3.54,3.54],color='#D9E1E8',lw=1)
txt(0.83,3.25,'论文 P1、P2 → 作者 A1',14,C['dark'])
txt(0.83,2.78,'论文 P3 → 作者 A2',14,C['dark'])
txt(0.83,2.31,'作者 A1、A2 → 机构 F1',14,C['dark'])
txt(0.83,1.72,'每条记录映射为节点；外键映射为边',13,C['blue'],'bold')

# Between panels: mapping arrow and explanation.
arrow(5.29,4.30,5.43,4.30,C['blue'],1.8,18)

# Center: one heterogeneous graph with two paths from P1.
P1=(6.05,5.29); A1=(7.12,5.18); P2=(8.95,6.10)
F1=(8.12,3.92); A2=(9.10,2.83); P3=(9.94,2.12)
connector(P1,A1,C['border'],2.1)           # common prefix
connector(A1,P2,C['orange_path'],3.2)       # shared author
connector(A1,F1,C['teal_path'],3.2)
connector(F1,A2,C['teal_path'],3.2)
connector(A2,P3,C['teal_path'],3.2)
for pt,k,lab,t in [(P1,'P','P1',True),(A1,'A','A1',False),(P2,'P','P2',False),
                   (F1,'F','F1',False),(A2,'A','A2',False),(P3,'P','P3',False)]:
    node(*pt,k,lab,target=t)
txt(6.05,6.30,'目标论文',13,'#E43934','bold','center')
txt(8.96,6.64,'共同作者',12,C['orange_path'],'bold','center')
txt(9.08,3.72,'同一机构',12,C['teal_path'],'bold','center')
ax.plot([5.83,10.11],[1.65,1.65],color='#D9E1E8',lw=1)
txt(7.99,1.48,'同一目标节点可经不同类型的路径接收信息',12.4,C['muted'],ha='center')
arrow(10.54,4.30,10.69,4.30,C['blue'],1.8,18)

# Right: type-level pattern and concrete instance paired by meaning.

def path_card(x,y,w,h,accent,bg,title,pattern,instance,meaning):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.13',
        facecolor=bg,edgecolor=accent,lw=1.35))
    txt(x+0.20,y+h-0.30,title,13.5,accent,'bold')
    txt(x+0.20,y+h-0.80,pattern,20,accent,'bold')
    txt(x+0.20,y+h-1.21,'实例：'+instance,12.6,C['dark'])
    txt(x+0.20,y+0.27,meaning,14,accent,'bold')

path_card(11.06,5.00,4.13,1.97,C['orange_path'],C['pale_orange'],
          '元路径 1 · 较直接的关系','P → A → P','P1 → A1 → P2','共享作者：P1 与 P2')
path_card(11.06,2.74,4.13,1.97,C['teal_path'],C['pale_teal'],
          '元路径 2 · 更高阶的关系','P → A → F → A → P','P1 → A1 → F1 → A2 → P3','同机构、不同作者：P1 与 P3')
ax.add_patch(FancyBboxPatch((11.05,1.49),4.16,0.94,
    boxstyle='round,pad=0.04,rounding_size=0.09',facecolor=C['light'],edgecolor='#D2DDE5',lw=1))
txt(11.27,2.13,'路径类型不同，关系语义也不同；',13.5,C['dark'],'bold')
txt(11.27,1.76,'应分别聚合，再学习任务相关的融合权重。',13.5,C['dark'],'bold')

# Takeaway and small source line.
ax.add_patch(FancyBboxPatch((0.53,0.36),14.97,0.58,
    boxstyle='round,pad=0.02,rounding_size=0.07',facecolor=C['blue'],edgecolor='none'))
txt(8.02,0.65,'元路径保留跨表关系的语义身份；MetaRTL 分路径聚合，再由 MetaAttn 按任务融合。',17,'white','bold','center')
txt(0.58,0.10,'概念依据：Sun et al., PathSim, PVLDB 2011；Dong et al., metapath2vec, KDD 2017；Wang et al., HAN, WWW 2019；Zhong et al., MetaRTL, 2026。',9.2,C['muted'])

svg=OUT/'元路径与关系表格语义对应.svg'
portable_svg=OUT/'元路径与关系表格语义对应-PPT兼容.svg'
png=OUT/'元路径与关系表格语义对应.png'
pdf=OUT/'元路径与关系表格语义对应.pdf'
fig.savefig(svg,transparent=True,metadata={'Creator':'MetaRTL-style relational meta-path diagram'})
mpl.rcParams['svg.fonttype']='path'
fig.savefig(portable_svg,transparent=True,metadata={'Creator':'MetaRTL-style relational meta-path diagram (outlined text)'})
fig.savefig(png,dpi=200,transparent=True)
fig.savefig(pdf,transparent=True)
plt.close(fig)
print(svg)
print(portable_svg)
print(png)
print(pdf)
