import hoomd
import matplotlib
import numpy as np
import itertools
import gsd.hoomd
import math

# Initialize system
N_particles = 10000
spacing = 20.0
K = math.ceil(N_particles**(1 / 2))
L = K * spacing
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = np.zeros((N_particles,3))
count = 0
for i in range(K):
    for j in range(K):
        position[count,0] = x[i]
        position[count,1] = x[j]
        count = count+1
        if(count == N_particles):
            break
    if(count == N_particles):
        break
print(position)

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * N_particles
snapshot.particles.mass = [1] * N_particles
snapshot.particles.diameter = [1] * N_particles
snapshot.configuration.box = [L, L, 0, 0, 0, 0]
snapshot.particles.types = ['A']
with gsd.hoomd.open(name='lattice.gsd', mode='wb') as f:
    f.append(snapshot)
