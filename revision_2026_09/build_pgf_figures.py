"""Write PGFPlots sources and data tables; all rendering is done by LaTeX."""
from pathlib import Path
import csv
A=Path(__file__).resolve().parent;O=A/'latex_figures';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((A/'summary/curves.csv').open()))
PRE=r'''\documentclass[tikz,border=2mm]{standalone}
\usepackage{amsmath}\usepackage{pgfplots}
\pgfplotsset{compat=1.18}\usepgfplotslibrary{groupplots,fillbetween}\usetikzlibrary{calc}
\definecolor{pers}{HTML}{222222}\definecolor{meanref}{HTML}{0072B2}
\definecolor{driftref}{HTML}{D55E00}\definecolor{arref}{HTML}{009E73}
\begin{document}\begin{tikzpicture}
'''
STYLE=r'''scale only axis,ymode=log,ymin=0.01,ymax=100,
axis lines=left,axis line style={line width=0.3pt,-},tick style={line width=0.3pt},
tick label style={font=\fontsize{7}{8}\selectfont},label style={font=\fontsize{7}{8}\selectfont},title style={font=\fontsize{7}{8}\selectfont},
minor ytick={},ytick={0.01,0.1,1,10,100},yticklabels={0.01,0.1,1,10,100},log ticks with fixed point,
unbounded coords=jump,legend style={font=\fontsize{7}{8}\selectfont,draw=none,fill=none,/tikz/every even column/.append style={column sep=2mm}},legend columns=-1'''
REFS=[('persistence','pers','Persistence'),('history_mean','meanref','History mean'),('drift','driftref','Drift'),('ar1','arref','AR(1)')]
def curve(sy,d,m,ref,col,idx,mult=1,legend=None):
 dat=sorted([r for r in rows if r['system']==sy and r['domain']==d and r['model']==m and r['reference']==ref and r['skill']],key=lambda r:float(r['severity']))
 name=f'data_{idx}.dat'
 with (O/name).open('w') as f:
  f.write('x skill lower upper\n')
  for r in dat:f.write(' '.join([str(float(r['severity'])*mult),r['skill'],r['ci_low'],r['ci_high']])+'\n')
 s=rf'''\addplot[name path=lo{idx},draw=none,forget plot] table[x=x,y=lower] {{{name}}};
\addplot[name path=hi{idx},draw=none,forget plot] table[x=x,y=upper] {{{name}}};
\addplot[{col},fill opacity=0.10,draw=none,forget plot] fill between[of=lo{idx} and hi{idx}];
\addplot[{col},line width=0.65pt,mark=*,mark size=0.8pt{',forget plot' if legend is None else ''}] table[x=x,y=skill] {{{name}}};
'''
 if legend:s+=rf'\addlegendentry{{{legend}}}'+'\n'
 return s

def ending(legend,cols):return r'\end{groupplot}'+'\n'+rf'\node[anchor=north] at ([yshift=-11mm]$(group c1r2.south)!0.5!(group c{cols}r2.south)$) {{\pgfplotslegendfromname{{{legend}}}}};'+'\n'+r'\end{tikzpicture}\end{document}'
panels=[('mbr','obs_noise','MBR: observation noise',1,1,'Noise scale (training SD)'),('tep','actuation_idv14','TEP: valve sticking',1,8,'Stiction multiplier'),('tep','regime_setpt18','TEP: setpoint shift',10,10,r'$\Delta T_r$ ($^\circ$C)')]
s=PRE+r'\begin{groupplot}[group style={group size=3 by 2,horizontal sep=12mm,vertical sep=14mm},width=49mm,height=34mm,'+STYLE+']\n'
for j,m in enumerate(['ridge','gru_s42']):
 for i,(sy,d,title,mult,xmax,xlab) in enumerate(panels):
  letter='abcdef'[j*3+i];op=[rf'title={{\textbf{{{letter}}} {title}}}',f'xmin=0,xmax={xmax}']
  if i==0:op.append(r'ylabel={'+('Ridge' if j==0 else 'GRU seed 42')+': reference/model RMSE}')
  else:op.append('yticklabels={}')
  if j==1:op.append('xlabel={'+xlab+'}')
  if i==0 and j==0:op.append('legend to name=sharedrefs')
  s+='\\nextgroupplot['+','.join(op)+']\n'+r'\addplot[gray,dashed,line width=0.4pt,forget plot] coordinates {(0,1) (100,1)};'+'\n'
  for k,(ref,c,l) in enumerate(REFS):s+=curve(sy,d,m,ref,c,f'ref{j}{i}{k}',mult,l if i==0 and j==0 else None)
(O/'reference_sensitivity.tex').write_text(s+ending('sharedrefs',3))
titles={'obs_bias':'Observation bias','obs_noise':'Observation noise','process':'Process input','actuation':'Actuation','process_idv1':'Process / IDV(1)','actuation_idv6':'Actuation / IDV(6)','dynamics_idv13':'Dynamics / IDV(13)','actuation_idv14':'Actuation / IDV(14)','regime_setpt18':'Regime / setpoint'}
for sy in ['mbr','tep']:
 domains=['obs_bias','obs_noise','process','actuation'] if sy=='mbr' else ['obs_bias','obs_noise','process_idv1','actuation_idv6','dynamics_idv13','actuation_idv14','regime_setpt18']
 cols=2 if sy=='mbr' else 4;w=76 if sy=='mbr' else 34
 s=PRE+rf'\begin{{groupplot}}[group style={{group size={cols} by 2,horizontal sep=12mm,vertical sep=14mm}},width={w}mm,height=32mm,'+STYLE+']\n'
 for i,d in enumerate(domains):
  lev=[float(r['severity']) for r in rows if r['system']==sy and r['domain']==d and r['skill']]
  op=[f'title={{{titles[d]}}}','xlabel={Operator severity}',f'xmin={min(lev)},xmax={max(lev)}']
  op+=['ylabel={Persistence/model RMSE}'] if i%cols==0 else ['yticklabels={}']
  if i==0:op.append(f'legend to name={sy}legend')
  s+='\\nextgroupplot['+','.join(op)+']\n'+r'\addplot[gray,dashed,line width=0.4pt,forget plot] coordinates {(0,1) (100,1)};'+'\n'
  for k,(m,c,l) in enumerate([('ridge','pers','Ridge'),('mlp_s42','meanref','MLP seed 42'),('gru_s42','driftref','GRU seed 42')]):s+=curve(sy,d,m,'persistence',c,f'{sy}{i}{k}',legend=l if i==0 else None)
 if sy=='tep':s+=r'\nextgroupplot[hide axis,xmin=0,xmax=1]'+'\n'
 (O/('skill_retention.tex' if sy=='mbr' else 'tep_retention_v2.tex')).write_text(s+ending(sy+'legend',cols))
print(O)
# Bound threshold segments to each axis before PGF transforms coordinates.
import re
for f in O.glob('*.tex'):
 parts=f.read_text().split('\\nextgroupplot[')
 for i in range(1,len(parts)):
  bound=re.search(r'xmax=([0-9.]+)',parts[i])
  if bound:parts[i]=parts[i].replace('(100,1)',f'({bound[1]},1)')
 f.write_text('\\nextgroupplot['.join(parts))
