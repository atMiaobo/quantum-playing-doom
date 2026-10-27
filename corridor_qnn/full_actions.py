"""Compact full-game macro actions; ordering is part of checkpoint schema."""
ACTIONS = (
    ('MOVE_FORWARD',), ('MOVE_BACKWARD',), ('TURN_LEFT',), ('TURN_RIGHT',),
    ('MOVE_LEFT',), ('MOVE_RIGHT',), ('ATTACK',), ('MOVE_FORWARD','ATTACK'),
    ('TURN_LEFT','ATTACK'), ('TURN_RIGHT','ATTACK'),
    ('MOVE_LEFT','ATTACK'), ('MOVE_RIGHT','ATTACK'),
    ('USE',), ('MOVE_FORWARD','USE'), ('MOVE_FORWARD','TURN_LEFT'),
    ('MOVE_FORWARD','TURN_RIGHT'), ('SELECT_WEAPON2',), ('SELECT_WEAPON3',),
    ('SELECT_WEAPON4',), ('SELECT_NEXT_WEAPON',), ('MOVE_BACKWARD','ATTACK'),
)
ACTION_NAMES = tuple('+'.join(a).lower() for a in ACTIONS)
BUTTONS = tuple(dict.fromkeys(b for a in ACTIONS for b in a))


def action_vectors(available):
    missing = set(BUTTONS) - set(available)
    if missing:
        raise ValueError(f'Full-game buttons missing: {sorted(missing)}')
    return [[int(b in a) for b in available] for a in ACTIONS]
