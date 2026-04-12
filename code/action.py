"""
action.py
=========
Action class for the two-player resource-and-combat MDP.

Replaces the plain strings ('P1_train_workers', 'P2_attack', etc.) used
throughout the codebase. Actions are immutable, hashable, and string-
compatible: hash(Action('P1', 'attack')) == hash('P1_attack') and
Action('P1', 'attack') == 'P1_attack', so Action objects and string literals
are interchangeable as dict keys without breaking existing code.

Fields: player ('P1' or 'P2'), type ('train_workers', 'train_marines', 'attack')

Class-level constants:
    Action.P1_TRAIN_WORKERS, Action.P1_TRAIN_MARINES, Action.P1_ATTACK
    Action.P2_TRAIN_WORKERS, Action.P2_TRAIN_MARINES, Action.P2_ATTACK
    Action.P1_ACTIONS  — list of all P1 actions
    Action.P2_ACTIONS  — list of all P2 actions
    Action.ALL         — list of all six actions
"""


class Action:
    __slots__ = ('player', 'type')

    def __init__(self, player: str, action_type: str):
        object.__setattr__(self, 'player', player)
        object.__setattr__(self, 'type', action_type)

    def __setattr__(self, name, value):
        raise AttributeError("Action is immutable")

    # ── String compatibility ──────────────────────────────────────────────────

    def __str__(self) -> str:
        return f'{self.player}_{self.type}'

    def __repr__(self) -> str:
        return f"Action('{self.player}', '{self.type}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, Action):
            return self.player == other.player and self.type == other.type
        if isinstance(other, str):
            return str(self) == other  # Action('P1','attack') == 'P1_attack'
        return NotImplemented

    def __hash__(self) -> int:
        # Same hash as the equivalent string so Action objects work as drop-in
        # dict keys in tables built with string keys (policies, Q-tables, etc.)
        return hash(str(self))


# ── Class-level constants (defined after class so they're Action instances) ───

Action.P1_TRAIN_WORKERS = Action('P1', 'train_workers')
Action.P1_TRAIN_MARINES = Action('P1', 'train_marines')
Action.P1_ATTACK        = Action('P1', 'attack')
Action.P2_TRAIN_WORKERS = Action('P2', 'train_workers')
Action.P2_TRAIN_MARINES = Action('P2', 'train_marines')
Action.P2_ATTACK        = Action('P2', 'attack')

Action.P1_ACTIONS = [Action.P1_TRAIN_WORKERS, Action.P1_TRAIN_MARINES, Action.P1_ATTACK]
Action.P2_ACTIONS = [Action.P2_TRAIN_WORKERS, Action.P2_TRAIN_MARINES, Action.P2_ATTACK]
Action.ALL        = Action.P1_ACTIONS + Action.P2_ACTIONS
