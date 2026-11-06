"""Particle Swarm Optimization (PSO) written from scratch with NumPy."""
import numpy as np


class PSO:
    """Maximises `fitness(position)`.

    bounds : list of (low, high) for each dimension
    Each particle moves with:
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)     (velocity update)
        x = x + v                                            (position update)
    """

    def __init__(self, fitness, bounds, n_particles=12, n_iter=15,
                 w=0.7, c1=1.5, c2=1.5, seed=0, callback=None):
        self.fitness = fitness
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.n, self.iters = n_particles, n_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng = np.random.default_rng(seed)
        self.callback = callback

    def run(self):
        d = len(self.lo)
        span = self.hi - self.lo
        vmax = 0.3 * span

        x = self.rng.uniform(self.lo, self.hi, (self.n, d))          # positions
        v = self.rng.uniform(-vmax, vmax, (self.n, d)) * 0.5         # velocities
        pbest = x.copy()
        pfit = np.array([self.fitness(p) for p in x])
        g = pfit.argmax()
        gbest, gfit = pbest[g].copy(), pfit[g]
        history = [float(gfit)]

        for it in range(self.iters):
            r1, r2 = self.rng.random((self.n, d)), self.rng.random((self.n, d))
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            v = np.clip(v, -vmax, vmax)
            x = np.clip(x + v, self.lo, self.hi)

            fit = np.array([self.fitness(p) for p in x])
            better = fit > pfit                                      # update personal bests
            pbest[better], pfit[better] = x[better], fit[better]
            g = pfit.argmax()
            if pfit[g] > gfit:                                       # update global best
                gbest, gfit = pbest[g].copy(), pfit[g]
            history.append(float(gfit))
            if self.callback:
                self.callback(it + 1, self.iters)
        return gbest, float(gfit), history
