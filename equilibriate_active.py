import hoomd
import numpy as np
import gsd.hoomd
import math

gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=1)
sim.create_state_from_gsd(filename='restart.gsd')
integrator = hoomd.md.Integrator(dt=0.00001,integrate_rotational_dof=True)
cell = hoomd.md.nlist.Cell(buffer=0.4)

#WCA potential
lj = hoomd.md.pair.LJ(nlist=cell, default_r_cut=2**(1.0/6.0), mode='shift')
lj.params[('A', 'A')] = dict(epsilon=40.0, sigma=1)
integrator.forces.append(lj)

#Integration settings
brownian = hoomd.md.methods.Brownian(kT=1.0, filter=hoomd.filter.All())
brownian.gamma.default = 1.0
brownian.gamma['A'] = 1.0
brownian.gamma_r['A'] = [1.0/3.0,1.0/3.0,1.0/3.0]
integrator.methods.append(brownian)
sim.operations.integrator = integrator

#Active force
active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
active.active_force['A'] = (40,0,0)
active.active_torque['A'] = (0,0,0)
integrator.forces.append(active)
rotational_diffusion_updater = active.create_diffusion_updater(trigger=1, rotational_diffusion=3.0)
sim.operations += rotational_diffusion_updater

#Thermodynamic properties
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
sim.run(0)

#GSD
gsd_writer = hoomd.write.GSD(filename='trajectory.gsd',trigger=hoomd.trigger.Periodic(10000),mode='wb')
gsd_writer_restart = hoomd.write.GSD(filename='restart.gsd',trigger=hoomd.trigger.Periodic(1000),mode='wb',truncate=True)
sim.operations.writers.append(gsd_writer)
sim.operations.writers.append(gsd_writer_restart)

#Logger
logger = hoomd.logging.Logger()
logger.add(thermodynamic_properties, quantities=['kinetic_energy','potential_energy','pressure','volume'])
logger.add(sim, quantities=['timestep','tps'])

#Measure swim pressure
class SwimPressure():

    def __init__(self,sim,active_force,gamma,gamma_r,position_old=None):
        self.sim = sim
        self.active_force = active_force
        self.count = 0
        if position_old is not None:
            self.count = 1
            self.position_old = position_old
        self.gamma = gamma
        self.gamma_r = gamma_r

    @property
    def GetSwimPressure0(self):
        with self.sim.state.cpu_local_snapshot as snap:
            with self.active_force.cpu_local_force_arrays as active_force:
                #Divide this by the area to get the true swim pressure, then can get to running
                return 0.5*np.sum(snap.particles.position*active_force.force)/snap.local_box.volume

    @property
    def GetSwimPressure1(self):
        with self.sim.state.cpu_local_snapshot as snap:
            rtag = snap.particles.rtag
            if self.count == 0:
                self.position_old = snap.particles.position[rtag]
            with self.active_force.cpu_local_force_arrays as active_force:
                #Divide this by the area to get the true swim pressure, then can get to running
                if self.count == 0:
                    self.count = 1
                    return 0.5*np.sum(snap.particles.position*active_force.force)/snap.local_box.volume
                else:
                    Lx = snap.local_box.Lx
                    Ly = snap.local_box.Ly
                    Lz = snap.local_box.Lz
                    box = np.array([Lx,Ly,Lz])
                    position_new = snap.particles.position[rtag] - self.position_old
                    position_new[:,0:2] = position_new[:,0:2]-np.rint(position_new[:,0:2]/box[0:2])*box[0:2]
                    position_new = position_new+self.position_old
                    self.position_old = position_new
                    return 0.5*np.sum(position_new*active_force.force[rtag])/snap.local_box.volume

    @property
    def GetSwimPressure2(self):
        with self.sim.state.cpu_local_snapshot as snap:
            with self.active_force.cpu_local_force_arrays as active_force:
                #Divide this by the area to get the true swim pressure, then can get to running
                return 0.5*self.gamma/self.gamma_r*np.sum(snap.particles.net_force*active_force.force)/snap.local_box.volume

    @property
    def GetSwimPressure3(self):
        with self.sim.state.cpu_local_snapshot as snap:
            with self.active_force.cpu_local_force_arrays as active_force:
                #Divide this by the area to get the true swim pressure, then can get to running
                return 0.5*self.gamma/self.gamma_r*np.sum(snap.particles.velocity*active_force.force)/snap.local_box.volume

swim_pressure = SwimPressure(sim,active,gamma=1.0,gamma_r=3.0)
logger[('SwimPressure', 'GetSwimPressure0')] = (swim_pressure, 'GetSwimPressure0', 'scalar')
logger[('SwimPressure', 'GetSwimPressure1')] = (swim_pressure, 'GetSwimPressure1', 'scalar')
logger[('SwimPressure', 'GetSwimPressure2')] = (swim_pressure, 'GetSwimPressure2', 'scalar')
logger[('SwimPressure', 'GetSwimPressure3')] = (swim_pressure, 'GetSwimPressure3', 'scalar')

gsd_writer_log = hoomd.write.GSD(filename='log.gsd',trigger=hoomd.trigger.Periodic(100),mode='wb',filter=hoomd.filter.Null(), log=logger)
sim.operations.writers.append(gsd_writer_log)

#Run for long time
sim.run(1e6)
np.savetxt("old_positions.txt", swim_pressure.position_old)
