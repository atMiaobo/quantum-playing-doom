"""Privileged static Doom-format IWAD geometry planner, shared by all policies.

Rasterized collision approximation, not an engine pathfinder. Doors are considered
potentially traversable. Locks, lifts, teleports and scripted specials are not solved.
No map-specific coordinates or action sequences are stored.
"""
from collections import deque
from pathlib import Path
import struct
import numpy as np
from PIL import Image, ImageDraw


def map_name(value, secret=False):
    value = str(value).upper()
    if len(value) != 5 or not value.startswith('MAP') or not value[3:].isdigit():
        raise ValueError('Map must be MAP01 through MAP30 (MAP31/32 with --secret-maps)')
    if not 1 <= int(value[3:]) <= (32 if secret else 30):
        raise ValueError('Map outside supported range')
    return value


def read_map(path, name):
    data = Path(path).read_bytes()
    magic, count, offset = struct.unpack_from('<4sii', data)
    if magic not in (b'IWAD', b'PWAD'):
        raise ValueError('Invalid WAD header')
    lumps = [struct.unpack_from('<ii8s', data, offset+16*i) for i in range(count)]
    names = [n.rstrip(b'\0').decode('ascii') for _, _, n in lumps]
    if name not in names:
        raise ValueError(f'{name} absent from {path}')
    start = names.index(name)
    found = {}
    for off, size, n in lumps[start+1:start+12]:
        found[n.rstrip(b'\0').decode('ascii')] = data[off:off+size]
    if 'BEHAVIOR' in found or 'TEXTMAP' in found:
        raise ValueError('Planner currently supports classic Doom-format maps only')
    def records(key, fmt):
        return list(struct.iter_unpack(fmt, found[key]))
    verts = records('VERTEXES', '<hh')
    lines = records('LINEDEFS', '<HHHHHHH')
    sides = records('SIDEDEFS', '<hh8s8s8sH')
    sectors = records('SECTORS', '<hh8s8shhh')
    return verts, lines, sides, sectors


class LegacyNavigator:
    cell = 32.0

    def __init__(self, path, name):
        verts, lines, sides, sectors = read_map(path, name)
        xy = np.asarray(verts)
        self.origin = xy.min(0)-96
        size = np.ceil((xy.max(0)-self.origin+96)/self.cell).astype(int)+1
        if np.prod(size) > 2_000_000:
            raise ValueError('Map too large for bounded navigation grid')
        image = Image.new('L', tuple(size), 0)
        draw = ImageDraw.Draw(image)
        self.exits = []
        self.interactions = []
        for v1, v2, flags, special, tag, right, left in lines:
            p, q = xy[v1], xy[v2]
            if special in (11, 51, 52, 124, 197, 198):
                self.exits.append((p+q)/2)
            if special:
                self.interactions.append((p+q)/2)
            solid = left == 65535 or right == 65535 or bool(flags & 1)
            if not solid:
                a, b = sectors[sides[right][-1]], sectors[sides[left][-1]]
                door = a[0] == a[1] or b[0] == b[1]
                solid = not door and (abs(a[0]-b[0]) > 24 or min(a[1],b[1])-max(a[0],b[0]) < 56)
            if solid:
                a, b = (p-self.origin)/self.cell, (q-self.origin)/self.cell
                draw.line([tuple(a), tuple(b)], fill=1, width=1)
        self.blocked = np.asarray(image).astype(bool)
        self.reset()

    def reset(self):
        self.visits = np.zeros(self.blocked.shape, dtype=np.int32)
        self.path = []
        self.reachable = 1
        self.calls = 0

    def grid(self, xy):
        x, y = np.rint((np.asarray(xy)-self.origin)/self.cell).astype(int)
        return int(x), int(y)

    def world(self, node):
        return np.asarray(node)*self.cell+self.origin

    def free(self, node):
        x,y = node
        return 0 <= y < self.blocked.shape[0] and 0 <= x < self.blocked.shape[1] and not self.blocked[y,x]

    def route(self, start):
        queue = deque([start]); parent = {start: None}; dist = {start: 0}
        while queue:
            x,y = queue.popleft()
            for node in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if node not in parent and self.free(node):
                    parent[node]=(x,y); dist[node]=dist[(x,y)]+1; queue.append(node)
        self.reachable = len(parent)
        # Approach the nearest reachable cell to an actual exit linedef.
        if self.exits:
            goal = min(parent, key=lambda n: min(np.linalg.norm(self.world(n)-e) for e in self.exits))
            if self.visits[goal[1],goal[0]] > 8:
                goal = max(parent, key=lambda n: dist[n]/(1+self.visits[n[1],n[0]]*10))
        else:
            goal = max(parent, key=lambda n: dist[n]/(1+self.visits[n[1],n[0]]*10))
        path=[]
        while goal != start:
            path.append(goal); goal=parent[goal]
        return path[::-1]

    def update(self, xy):
        node = self.grid(xy)
        if 0 <= node[1] < self.visits.shape[0] and 0 <= node[0] < self.visits.shape[1]:
            self.visits[node[1],node[0]] += 1
        if not self.path or self.calls % 32 == 0 or np.linalg.norm(self.world(self.path[0])-xy)>160:
            self.path = self.route(node)
        self.calls += 1
        while len(self.path)>1 and np.linalg.norm(self.world(self.path[0])-xy)<45:
            self.path.pop(0)
        target = self.world(self.path[0]) if self.path else np.asarray(xy)
        near_use = min((np.linalg.norm(p-xy) for p in self.interactions), default=1e6)
        return target, near_use


class Navigator(LegacyNavigator):
    """Player-clearance grid, front-side use goals and local collision memory.

    Static IWAD information is privileged, as in the original planner. Closed
    doors are optimistic portals; locked doors, lifts and teleports may still
    require exploration. No map coordinates or recorded action sequences.
    """
    cell = 16.0
    radius = 18.0
    use_specials = {1, 11, 26, 27, 28, 29, 31, 32, 33, 34, 42, 50, 51,
                    61, 63, 103, 111, 112, 113, 114, 115, 116, 117, 118,
                    133, 134, 135, 136, 137}

    def __init__(self, path, name):
        verts, lines, sides, sectors = read_map(path, name)
        xy = np.asarray(verts, dtype=float)
        self.origin = xy.min(0)-96
        size = np.ceil((xy.max(0)-self.origin+96)/self.cell).astype(int)+1
        if np.prod(size)>2_000_000: raise ValueError('Map too large for bounded navigation grid')
        self.blocked = np.zeros(tuple(size[::-1]), dtype=bool)
        self.exits=[]; self.interactions=[]; self.use_lines=[]; self.walls=[]
        self.exit_approaches=[]
        # Doom's actor footprint can span a narrow depression. Treating each
        # bank as a separate impassable step incorrectly disconnects bridges.
        bounds={}; neighbors={}
        for v1,v2,flags,special,tag,right,left in lines:
            ids=[sides[z][-1] for z in (right,left) if z!=65535]
            for sid in ids:
                bounds.setdefault(sid,[]).extend((xy[v1],xy[v2]))
                neighbors.setdefault(sid,set()).update(set(ids)-{sid})
        self.spanned_sectors={}
        for sid,points in bounds.items():
            extent=np.ptp(points,axis=0)
            adjacent=[sectors[i][0] for i in neighbors[sid]]
            if adjacent and min(extent)<2*16 and max(adjacent)-min(adjacent)<=24 and min(adjacent)-sectors[sid][0]>24:
                self.spanned_sectors[sid]=min(adjacent)
        for v1,v2,flags,special,tag,right,left in lines:
            p,q=xy[v1],xy[v2]; midpoint=(p+q)/2
            normal=np.array([q[1]-p[1],p[0]-q[0]])/max(np.linalg.norm(q-p),1)
            if special in (11,51,52,124,197,198):
                self.exits.append(midpoint)
                # Doom switch activation uses the front (right) side of a linedef.
                self.exit_approaches.append((midpoint+normal*40,midpoint,normal))
            if special in self.use_specials:
                self.interactions.append(midpoint); self.use_lines.append((p,q,midpoint))
            solid=left==65535 or right==65535 or bool(flags&1)
            if not solid:
                ra,lb=sides[right][-1],sides[left][-1]
                a,b=list(sectors[ra]),list(sectors[lb])
                a[0]=self.spanned_sectors.get(ra,a[0]); b[0]=self.spanned_sectors.get(lb,b[0])
                door=a[0]==a[1] or b[0]==b[1]
                solid=not door and (abs(a[0]-b[0])>24 or min(a[1],b[1])-max(a[0],b[0])<56)
            if solid:
                self.walls.append((p,q))
                self._raster_segment(p,q,self.radius)
        self.reset()

    def _raster_segment(self,p,q,radius):
        lo=np.maximum(0,np.floor((np.minimum(p,q)-radius-self.origin)/self.cell).astype(int))
        hi=np.minimum(np.array(self.blocked.shape[::-1])-1,np.ceil((np.maximum(p,q)+radius-self.origin)/self.cell).astype(int))
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        pts=np.stack((xx,yy),axis=-1)*self.cell+self.origin
        delta=q-p; t=np.clip(np.sum((pts-p)*delta,axis=-1)/max(np.dot(delta,delta),1),0,1)
        hit=np.linalg.norm(pts-(p+t[...,None]*delta),axis=-1)<radius
        self.blocked[lo[1]:hi[1]+1,lo[0]:hi[0]+1]|=hit

    def reset(self):
        super().reset()
        self.recent=deque(maxlen=12); self.temporary={}; self.goal=None
        self.mode='route'; self.replans=0; self.collision_events=0

    def free(self,node):
        return super().free(node) and self.temporary.get(node,0)<=self.calls

    def clear(self,a,b):
        # Sample at half-cell spacing; also used to avoid cutting obstacle corners.
        n=max(2,int(np.linalg.norm(np.asarray(b)-a)/(self.cell/2))+1)
        return all(self.free(self.grid(p)) for p in np.linspace(a,b,n))

    def route(self,start):
        if not self.free(start):
            candidates=[(start[0]+x,start[1]+y) for x in range(-3,4) for y in range(-3,4)]
            good=[n for n in candidates if self.free(n)]
            if not good: return []
            start=min(good,key=lambda n:np.linalg.norm(np.asarray(n)-start))
        queue=deque([start]); parent={start:None}; dist={start:0}
        while queue:
            x,y=queue.popleft()
            for node in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if node not in parent and self.free(node):
                    parent[node]=(x,y); dist[node]=dist[(x,y)]+1; queue.append(node)
        self.reachable=len(parent); self.replans+=1
        nodes=list(parent); points=np.array(nodes)*self.cell+self.origin
        choices=[]
        for approach,mid,normal in self.exit_approaches:
            distance=np.linalg.norm(points-approach,axis=1)
            front=(points-mid)@normal>20
            for i in np.flatnonzero(front & (distance<32)):
                choices.append((dist[nodes[i]]+distance[i]/self.cell,nodes[i],mid))
        if choices:
            _,goal,self.goal=min(choices,key=lambda x:x[0]); self.mode='exit'
        else:
            # Favor unvisited reachable regions, retaining travel-cost pressure.
            goal=max(parent,key=lambda n:min(dist[n],40)/(1+self.visits[n[1],n[0]]*8))
            self.goal=None; self.mode='explore'
        path=[]
        while goal!=start:
            path.append(goal); goal=parent[goal]
        return path[::-1]

    def update(self,xy,yaw=None,last_action=-1,enemy_visible=False):
        xy=np.asarray(xy); node=self.grid(xy)
        self.calls+=1
        if 0<=node[1]<self.visits.shape[0] and 0<=node[0]<self.visits.shape[1]:
            self.visits[node[1],node[0]]+=1
        self.recent.append((xy.copy(),last_action))
        collision=False
        if len(self.recent)==12 and not enemy_visible and yaw is not None:
            moving=sum(a in (0,7,13,14,15) for _,a in self.recent)
            if moving>=7 and np.linalg.norm(xy-self.recent[0][0])<10:
                ahead=xy+32*np.array([np.cos(np.radians(yaw)),np.sin(np.radians(yaw))])
                # Temporarily reject a failed edge and seek an alternate approach.
                self.temporary[self.grid(ahead)]=self.calls+96
                self.recent.clear(); self.path=[]; collision=True; self.collision_events+=1
        if not self.path or self.calls%64==1 or np.linalg.norm(self.world(self.path[0])-xy)>128 or collision:
            self.path=self.route(node)
        while len(self.path)>1 and np.linalg.norm(self.world(self.path[0])-xy)<20:
            self.path.pop(0)
        target=self.world(self.path[0]) if self.path else xy.copy()
        # Bounded line-of-sight lookahead avoids grid zigzags and corner cutting.
        for waypoint in self.path[:8]:
            point=self.world(waypoint)
            if np.linalg.norm(point-xy)>96 or not self.clear(xy,point): break
            target=point
        if self.goal is not None and np.linalg.norm(xy-self.goal)<72:
            target=self.goal.copy()
        near=1e6
        for p,q,mid in self.use_lines:
            d=np.linalg.norm(mid-xy)
            # Only use interactions ahead of the route, rather than arbitrary specials.
            if d<80 and np.dot(mid-xy,target-xy)>0:
                delta=q-p; t=np.clip(np.dot(xy-p,delta)/max(np.dot(delta,delta),1),0,1)
                near=min(near,np.linalg.norm(xy-(p+t*delta)))
        return target,near
