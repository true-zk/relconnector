from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
D=Path(__file__).resolve().parent
out=Document();sec=out.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.8);sec.bottom_margin=Cm(1.8);sec.left_margin=Cm(2);sec.right_margin=Cm(2)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
 s=out.styles[name];s.font.name='Noto Sans CJK SC';s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
 s.font.color.rgb=RGBColor.from_string('1D2D42')
normal=out.styles['Normal'];normal.font.size=Pt(10.5);normal.paragraph_format.line_spacing=1.28;normal.paragraph_format.space_after=Pt(7)
for name,size in [('Title',22),('Heading 1',16),('Heading 2',12.5),('Heading 3',11.5)]:out.styles[name].font.size=Pt(size)
out.styles['Caption'].font.size=Pt(9);out.styles['Caption'].font.color.rgb=RGBColor.from_string('516477')
# footer page number
p=sec.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
p.add_run('RelConnector  ·  ')
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');p._p.append(fld)
def hyperlink(p,label,url):
 h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True))
 r=OxmlElement('w:r');pr=OxmlElement('w:rPr');c=OxmlElement('w:color');c.set(qn('w:val'),'296994');pr.append(c);r.append(pr);t=OxmlElement('w:t');t.text=label;r.append(t);h.append(r);p._p.append(h)
def inline(p,text):
 pattern=r'\[([^\]]+)\]\(([^)]+)\)'
 pos=0
 for m in re.finditer(pattern,text):
  p.add_run(text[pos:m.start()].replace('`',''))
  target=m.group(2)
  if not target.startswith('http'):target=(D/target).resolve().as_uri()
  hyperlink(p,('['+m.group(1)+']') if re.fullmatch(r'S?\d+',m.group(1)) else m.group(1),target);pos=m.end()
 p.add_run(text[pos:].replace('`',''))
lines=(D/'manuscript.md').read_text().splitlines();i=0;in_refs=False
while i<len(lines):
 line=lines[i].strip();i+=1
 if not line:continue
 if line.startswith('# '):out.add_heading(line[2:],0);continue
 if line.startswith('## '):
  if line.startswith('## 参考文献'):in_refs=True
  p=out.add_heading(line[3:],1)
  continue
 if line.startswith('### '):out.add_heading(line[4:],2);continue
 if line.startswith('!['):
  path=re.search(r'\]\(([^)]+)\)',line).group(1)
  p=out.add_paragraph();p.paragraph_format.keep_with_next=True;p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  p.add_run().add_picture(str(D/path),width=Cm(17));continue
 if line.startswith('|'):
  rows=[line]
  while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
  cells=[[c.strip() for c in r.strip('|').split('|')] for r in rows if not re.match(r'^\|\s*[-:]+',r)]
  table=out.add_table(rows=0,cols=len(cells[0]));table.style='Light Shading Accent 1'
  for rid,row in enumerate(cells):
   cs=table.add_row().cells
   for c,v in zip(cs,row):
    inline(c.paragraphs[0],v)
    for run in c.paragraphs[0].runs:run.font.size=Pt(8);run.bold=rid==0
    c.paragraphs[0].paragraph_format.line_spacing=1.1
   if rid==0:
    e=OxmlElement('w:tblHeader');table.rows[-1]._tr.get_or_add_trPr().append(e)
   cant=OxmlElement('w:cantSplit');table.rows[-1]._tr.get_or_add_trPr().append(cant)
  continue
 p=out.add_paragraph(style='Caption' if re.match(r'^(图|表) \d+\.',line) else 'Normal');inline(p,line)
 if in_refs:
  p.paragraph_format.line_spacing=1.1;p.paragraph_format.space_after=Pt(5)
  for run in p.runs:run.font.size=Pt(9)
out.core_properties.title='RelConnector：背景、动机与方法'
out.core_properties.author='RelConnector'
out.core_properties.subject='基于历史测量与当前实现的论文片段'
out.save(D/'RelConnector-背景动机与方法.docx')
print('Exported DOCX; manuscript characters:',len((D/'manuscript.md').read_text()))
