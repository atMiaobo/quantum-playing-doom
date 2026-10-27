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


class Navigator:
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
