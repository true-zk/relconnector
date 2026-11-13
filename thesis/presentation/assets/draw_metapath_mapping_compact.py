"""Compact paper figure: relational rows -> heterogeneous graph -> meta-paths."""
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager
from fontTools.ttLib import TTFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon, FancyArrowPatch
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
for weight in ('Regular','Bold'):
    src=Path(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc')
    dst=Path(f'/tmp/codex-NotoSansCJKSC-{weight}.otf')
    if not dst.exists():TTFont(str(src),fontNumber=2).save(str(dst))
    font_manager.fontManager.addfont(str(dst))
mpl.rcParams['font.family']='Noto Sans CJK SC'
mpl.rcParams['svg.fonttype']='none'
mpl.rcParams['pdf.fonttype']=42
P='#F18C45'; A='#537BC8'; F='#3FC2B7'; DARK='#29323C'; GREY='#89939D'; LINE='#C7D0D8'; O='#D97422'; T='#168F87'; RED='#DC3531'
fig,ax=plt.subplots(figsize=(13.5,3.7),dpi=240)
fig.patch.set_alpha(0);ax.patch.set_alpha(0)
ax.set_xlim(0,13.5);ax.set_ylim(0,3.7);ax.axis('off')
plt.subplots_adjust(left=0,right=1,top=1,bottom=0)

def txt(x,y,t,s=13,c=DARK,b=False,ha='left'):
    ax.text(x,y,t,fontsize=s,color=c,fontweight='bold' if b else 'normal',ha=ha,va='center',fontfamily='Noto Sans CJK SC',zorder=10)

def box(x,y,w,h,fill='white',edge=LINE,lw=1):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=lw))

def arr(a,b,color=DARK,lw=1.5,scale=15):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=scale,lw=lw,color=color,shrinkA=0,shrinkB=0))

def node(x,y,kind,label,r=.25,target=False):
    if kind=='P':
        ax.add_patch(Circle((x,y),r,facecolor=P,edgecolor=RED if target else DARK,lw=2 if target else 1.3,zorder=5))
    elif kind=='A':
        ax.add_patch(Rectangle((x-r,y-r),2*r,2*r,facecolor=A,edgecolor=DARK,lw=1.3,zorder=5))
    else:
        ax.add_patch(RegularPolygon((x,y),3,radius=r*1.25,orientation=0,facecolor=F,edgecolor=DARK,lw=1.3,zorder=5))
    txt(x,y,label,12,'white',True,'center')

def table(x,top,w,title,fields,rows,color):
    rh=.48;head=.46;col=.39
    h=head+col+len(rows)*rh
    y=top-h
    box(x,y,w,h,'white',DARK,1.1)
    box(x,top-head,w,head,color,DARK,1.1)
    txt(x+w/2,top-head/2,title,14,'white',True,'center')
    box(x,top-head-col,w,col,'#F1F4F7',LINE,.5)
    txt(x+w/2,top-head-col/2,fields,10.8,'#5A6671',False,'center')
    for i,row in enumerate(rows):
        yy=top-head-col-(i+1)*rh
        box(x,yy,w,rh,'#FAFBFC' if i%2 else 'white',LINE,.45)
        txt(x+w/2,yy+rh/2,row,13,DARK,False,'center')
    return y

# Small labels identify the three representations; the figure has no slide frame.
txt(.18,3.53,'关系表格',12,DARK,True)
txt(5.48,3.53,'关系实例图',12,DARK,True)
txt(9.91,3.53,'元路径',12,DARK,True)

# Three tables with compact row-level examples.
table(.18,3.37,1.84,'论文 P','pid   author_id',['P1      A1','P2      A1','P3      A2'],P)
table(2.34,3.37,1.50,'作者 A','aid  firm_id',['A1     F1','A2     F1'],A)
table(4.16,3.37,.84,'机构 F','fid',['F1'],F)
arr((2.05,1.86),(2.28,1.86),DARK,1.2,11)
arr((3.87,1.86),(4.10,1.86),DARK,1.2,11)
txt(2.16,1.56,'FK',9,GREY,False,'center')
txt(3.98,1.56,'FK',9,GREY,False,'center')
txt(.22,.67,'P1、P2 → A1；P3 → A2',11,DARK)
txt(.22,.29,'A1、A2 → F1',11,DARK)
arr((5.10,1.76),(5.39,1.76),'#626E79',1.6,16)

# All rows become nodes; foreign keys become graph edges.
P1=(5.88,2.37);A1=(6.86,2.33);P2=(8.27,3.13)
F1=(7.53,1.50);A2=(8.48,.91);P3=(9.32,.38)
for a,b,c,lw in [(P1,A1,DARK,2.1),(A1,P2,O,2.8),(A1,F1,T,2.8),(F1,A2,T,2.8),(A2,P3,T,2.8)]:
    ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],lw=lw,color=c,zorder=2,solid_capstyle='round'))
for pt,k,l,t in [(P1,'P','P1',True),(A1,'A','A1',False),(P2,'P','P2',False),(F1,'F','F1',False),(A2,'A','A2',False),(P3,'P','P3',False)]:
    node(*pt,k,l,target=t)
arr((9.57,1.76),(9.82,1.76),'#626E79',1.6,16)

# Schema-level paths, each paired with an instance path.
txt(9.92,3.12,'P — A — P',17,O,True)
txt(9.92,2.72,'P1 — A1 — P2',12,DARK)
txt(9.92,2.32,'共享作者',11.5,O,True)
ax.plot([9.91,13.32],[1.93,1.93],color=LINE,lw=1)
txt(9.92,1.55,'P — A — F — A — P',16,T,True)
txt(9.92,1.15,'P1 — A1 — F1 — A2 — P3',11.7,DARK)
txt(9.92,.75,'同机构、不同作者',11.5,T,True)

base=OUT/'元路径与关系表格映射-论文插图'
fig.savefig(str(base)+'.svg',transparent=True)
mpl.rcParams['svg.fonttype']='path'
fig.savefig(str(base)+'-PPT兼容.svg',transparent=True)
fig.savefig(str(base)+'.png',dpi=240,transparent=True)
fig.savefig(str(base)+'.pdf',transparent=True)
plt.close(fig)
print(base)
