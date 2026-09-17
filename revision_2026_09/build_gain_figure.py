from pathlib import Path
import json, ast
A=Path(__file__).resolve().parent;O=A/'latex_figures';O.mkdir(exist_ok=True)
r=json.loads((A/'summary/gain_statistics.json').read_text())
gs=r['gains'];mins=r['model_minima'];ds=list(mins);ms=['ridge','mlp','gru']
PRE=next(ast.literal_eval(n.value) for n in ast.parse((A/'build_pgf_figures.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PRE' for t in n.targets))
s=PRE+r'\begin{groupplot}[group style={group size=5 by 1,horizontal sep=7mm},scale only axis,width=28mm,height=35mm,xmin=0.69,xmax=1.01,x dir=reverse,ymin=0.5,ymax=2.4,axis lines=left,axis line style={-},xtick={0.7,0.8,0.9,1},tick label style={font=\fontsize{7}{8}\selectfont},label style={font=\fontsize{7}{8}\selectfont},title style={font=\fontsize{7}{8}\selectfont},legend style={font=\fontsize{7}{8}\selectfont,draw=none},legend columns=3]'+'\n'
for i,(dom,title) in enumerate(zip(ds,['Process','Actuation','Dynamics','Obs. bias','Obs. noise'])):
 s+=r'\nextgroupplot[title={'+title+r'},xlabel={Gain}'+(r',ylabel={Minimum skill},legend to name=gainlegend' if i==0 else ',yticklabels={}')+']\n'
 s+=r'\addplot[gray,dashed,forget plot] coordinates {(0.7,1) (1,1)};'+'\n'
 for m,col,label in zip(ms,['pers','meanref','driftref'],['Ridge','MLP seed 42','GRU seed 42']):
  fn=f'gain_{dom}_{m}.dat';(O/fn).write_text('gain skill\n'+'\n'.join(f'{g} {v}' for g,v in zip(gs,mins[dom][m]))+'\n')
  s+=rf'\addplot[{col},line width=0.65pt,mark=*,mark size=0.8pt'+(',forget plot' if i else '')+rf'] table[x=gain,y=skill] {{{fn}}};'+'\n'
  if i==0:s+=r'\addlegendentry{'+label+'}\n'
s+=r'\end{groupplot}\node[anchor=north] at ([yshift=-11mm]$(group c1r1.south)!0.5!(group c5r1.south)$) {\pgfplotslegendfromname{gainlegend}};\end{tikzpicture}\end{document}'
(O/'gain_sweep.tex').write_text(s)
