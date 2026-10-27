"""Reactive waypoint/combat controller, not a learned policy or recorded solution."""
import math
import numpy as np
from .full_environment import BASE_FEATURES


def teacher_action(features):
    f=np.asarray(features)
    wb=math.atan2(f[9],f[8]); eb=math.atan2(f[13],f[12])
    last=int(np.argmax(f[len(BASE_FEATURES):])) if f[len(BASE_FEATURES):].sum() else -1
    if f[31]>.5 and f[3]*9<3 and last!=17: return 17
    # Combat uses visible engine monster labels, not through-wall object positions.
    if f[15]>.5 and f[1]>0:
        if eb>.12: return 8
        if eb<-.12: return 9
        return 7 if f[14]>.35 and f[22]<.5 else 6
    # Recovery is observable to the student via recent movement and last action.
    if f[22]>.5:
        if last in (0,7,13,14,15): return 12
        if last==12: return 1
        if last==1: return 4
        if last==4: return 2
    if wb>.18: return 2
    if wb<-.18: return 3
    if f[11]<.85 and last!=13: return 13
    return 0
