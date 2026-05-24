"""Discrete-time Markov chain regulation signal over S_r = {-4, -1, 1, 5}.

Transition matrix is not specified in the paper. We construct a zero-mean,
correlated DTMC: stay with prob 0.5, otherwise pick neighbor with bias toward
mean reversion. Same for all experiments.
"""
from __future__ import annotations

import numpy as np


class MarkovSignal:
    def __init__(self, states: tuple[int, ...] = (-4, -1, 1, 5), seed: int = 0):
        self.states = np.array(states, dtype=int)
        self.K = len(states)
        self.P = self._build_transition()
        self.rng = np.random.default_rng(seed)
        self.idx = self.K // 2  # start near zero

    def _build_transition(self) -> np.ndarray:
        # mean-reverting random walk on the index, with stickiness
        K = self.K
        P = np.zeros((K, K))
        for i in range(K):
            P[i, i] = 0.5
            # neighbors
            if i == 0:
                P[i, 1] = 0.5
            elif i == K - 1:
                P[i, K - 2] = 0.5
            else:
                # bias toward center
                center = (K - 1) / 2
                if i < center:
                    P[i, i + 1] = 0.30
                    P[i, i - 1] = 0.20
                elif i > center:
                    P[i, i - 1] = 0.30
                    P[i, i + 1] = 0.20
                else:
                    P[i, i + 1] = 0.25
                    P[i, i - 1] = 0.25
        # ensure rows sum to 1
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def reset(self) -> int:
        self.idx = self.K // 2
        return int(self.states[self.idx])

    def step(self) -> int:
        self.idx = self.rng.choice(self.K, p=self.P[self.idx])
        return int(self.states[self.idx])

    @property
    def support(self) -> np.ndarray:
        return self.states
