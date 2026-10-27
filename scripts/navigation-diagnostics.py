"""Offline clearance/trajectory overlay from saved real-game traces."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from corridor_qnn.full_navigation import Navigator,LegacyNavigator,read_map
root=Path('outputs/map01_navigation'); root.mkdir(parents=True,exist_ok=True)
wad=root/'diagnostic_iwad.wad'
if not wad.exists():
 import vizdoom
 wad=Path(vizdoom.install_path)/'freedoom2.wad'
for cls in (LegacyNavigator,Navigator):
 n=cls(wad,'MAP01'); n.update(np.array([-192.,-192.]))
 print(cls.__name__,'shape',n.blocked.shape,'reachable',n.reachable,'mode',getattr(n,'mode',None),'goal',getattr(n,'goal',None),'end',n.world(n.path[-1]) if n.path else None)
 scale=4 if cls is LegacyNavigator else 2
 im=Image.fromarray(np.where(n.blocked[...,None],np.array([160,160,160]),np.array([20,20,20])).astype('uint8')).resize((n.blocked.shape[1]*scale,n.blocked.shape[0]*scale))
 dr=ImageDraw.Draw(im)
 def pt(x): return tuple(((np.array(x)-n.origin)/n.cell*scale).astype(int))
 for e in n.exits:
  a,b=pt(e); dr.ellipse((a-6,b-6,a+6,b+6),fill='yellow'); dr.text((a+8,b),'EXIT',fill='yellow')
 for path in (root/'diagnostic_legacy_teacher/traces/teacher_600.json',root/'iteration1_teacher/traces/teacher_600.json'):
  if path.exists():
   t=json.loads(path.read_text())['trace']; color='cyan' if 'legacy' in str(path) else 'red'
   dr.line([pt(x['position']) for x in t],fill=color,width=2)
   for x in t[::100]: dr.text(pt(x['position']),str(x['step']),fill=color)
 im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(root/f'geometry_{cls.__name__}.png')
