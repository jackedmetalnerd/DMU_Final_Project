"""
dqn.py
======
Deep Q-Network (DQN) solver for the two-player resource-and-combat MDP.

Ported from code/hw5_p3_working.jl (Julia/Flux), adapted to this project's
7-dimensional game state and 3 P1 actions.

Architecture
------------
- Network: Linear(7, 256) → ReLU → Linear(256, 3)  [one Q-value per P1 action]
- Replay buffer: circular buffer, capacity 100,000
- Target network: frozen copy of Q, updated every target_update_freq steps
- Loss: MSE between predicted Q(s,a) and Bellman target
- Optimizer: Adam

State Encoding
--------------
State (W1, M1, R1, W2, M2, R2, terminal) → float32 tensor of shape (7,)
  [W1/10, M1/10, R1/10, W2/10, M2/10, R2/10, float(terminal)]
All values normalized to [0, 1].
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game_env import GameEnv
from state import State
from policy import DictPolicy
from solver import Solver
from policies import alternating_training_attack


# ── Neural network ────────────────────────────────────────────────────────────

def _build_network(input_size: int = 7, hidden_size: int = 256, output_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


# ── Replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Circular replay buffer storing (s, a_idx, r, s', done) 5-tuples."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer   = [None] * capacity
        self._write    = 0
        self._size     = 0

    def push(self, s_tensor, action_idx: int, reward: float, sp_tensor, done: bool):
        self._buffer[self._write] = (s_tensor, action_idx, reward, sp_tensor, done)
        self._write = (self._write + 1) % self._capacity
        self._size  = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer[:self._size], batch_size)
        s_batch, a_batch, r_batch, sp_batch, done_batch = zip(*batch)
        return (
            torch.stack(s_batch),
            torch.tensor(a_batch,  dtype=torch.long),
            torch.tensor(r_batch,  dtype=torch.float32),
            torch.stack(sp_batch),
            torch.tensor(done_batch, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return self._size


# ── DQNSolver ─────────────────────────────────────────────────────────────────

class DQNSolver(Solver):
    """DQN solver for a GameEnv.

    Hyperparameters mirror the Julia implementation (hw5_p3_working.jl),
    adapted for this project's 7-dim state and 3 P1 actions.
    """

    def __init__(
        self,
        env: GameEnv,
        gamma: float           = 0.95,
        epsilon: float         = 0.5,
        epsilon_decay: float   = 0.999,
        epsilon_min: float     = 0.01,
        lr: float              = 0.0005,
        batch_size: int        = 50,
        buffer_capacity: int   = 100_000,
        target_update_freq: int = 50,
    ):
        super().__init__(env)
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        self.epsilon_min        = epsilon_min
        self.lr                 = lr
        self.batch_size         = batch_size
        self.buffer_capacity    = buffer_capacity
        self.target_update_freq = target_update_freq

        self._n_actions = len(env.A)
        self._q_net     = _build_network(output_size=self._n_actions)
        self._q_target  = copy.deepcopy(self._q_net)
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=lr)
        self._buffer    = _ReplayBuffer(buffer_capacity)
        self.v0_history = []
        self._s0_tensor = None  #set on first solve() call

    # ── Public interface ──────────────────────────────────────────────────────

    def solve(self, n_episodes: int = 5000) -> DictPolicy:
        """Train for n_episodes; return a greedy DictPolicy over all states."""
        env           = self.env
        epsilon       = self.epsilon
        global_steps  = 0
        self._s0_tensor = self._encode(env.initial_state)


        for episode in range(1, n_episodes + 1):
            env.reset()
            s = env.observe()

            while not s.terminal:
                s_tensor = self._encode(s)
                a, a_idx = self._epsilon_greedy(s_tensor, epsilon)
                reward   = env.act(a)
                sp       = env.observe()
                sp_tensor = self._encode(sp)

                self._buffer.push(s_tensor, a_idx, reward, sp_tensor, sp.terminal)
                global_steps += 1

                if len(self._buffer) >= self.batch_size:
                    self._update()

                if global_steps % self.target_update_freq == 0:
                    self._q_target.load_state_dict(self._q_net.state_dict())

                s = sp

            with torch.no_grad():
                v0 = self._q_net(self._s0_tensor).max().item()
            self.v0_history.append(v0)

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            
            if n_episodes >= 10_000: #cleaner output for large runs
                if episode % 1000 == 0:
                    print(f"Episode {episode}/{n_episodes}  ε={epsilon:.4f}")
            else:
                if episode % 100 == 0:
                    print(f"Episode {episode}/{n_episodes}  ε={epsilon:.4f}")

        return self._build_policy()

    def policy(self, s=None):
        """Return greedy action for state s, or a DictPolicy over all states.

        Called with no arguments: returns a DictPolicy built from Q-network.
        Called with a state: returns the greedy Action for that state.
        """
        if s is None:
            return self._build_policy()
        with torch.no_grad():
            q_vals = self._q_net(self._encode(s))
        return self.env.A[q_vals.argmax().item()]

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _encode(s: State) -> torch.Tensor:
        """Convert a State to a normalized float32 tensor of shape (7,)."""
        return torch.tensor(
            [s.W1 / 10.0, s.M1 / 10.0, s.R1 / 10.0,
             s.W2 / 10.0, s.M2 / 10.0, s.R2 / 10.0,
             float(s.terminal)],
            dtype=torch.float32,
        )

    def _epsilon_greedy(self, s_tensor: torch.Tensor, epsilon: float):
        """Return (Action, action_index) using ε-greedy exploration."""
        if random.random() < epsilon:
            idx = random.randrange(self._n_actions)
        else:
            with torch.no_grad():
                idx = self._q_net(s_tensor).argmax().item()
        return self.env.A[idx], idx

    def _update(self):
        """Sample a mini-batch and perform one gradient step."""
        s_batch, a_batch, r_batch, sp_batch, done_batch = self._buffer.sample(self.batch_size)

        # Predicted Q(s, a) for the taken action
        q_pred = self._q_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

        # Bellman target: r  if done,  r + γ · max_a' Q_target(s', a')  otherwise
        with torch.no_grad():
            max_q_next = self._q_target(sp_batch).max(dim=1).values
            q_target   = torch.where(done_batch, r_batch, r_batch + self.gamma * max_q_next)

        loss = F.mse_loss(q_pred, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _build_policy(self) -> DictPolicy:
        """Evaluate the Q-network on every state and return a DictPolicy."""
        policy_dict = {}
        self._q_net.eval()
        with torch.no_grad():
            for s in self.env.S:
                idx = self._q_net(self._encode(s)).argmax().item()
                policy_dict[s] = self.env.A[idx]
        self._q_net.train()
        return DictPolicy(policy_dict)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
    env    = GameEnv(opponent_policy=alternating_training_attack, initial_state=s_init)

    agent = DQNSolver(env)
    print("Training DQN agent...")
    π_star = agent.solve(n_episodes=2000)
    print("Training complete.")

    print("\nSimulating games with learned policy...")
    for _ in range(5):
        env.simulate(π_star, label='DQN')
