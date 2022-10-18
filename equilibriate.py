import hoomd
import numpy as np
import gsd.hoomd
import math

gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=1)
sim.create_state_from_gsd(filename='lattice.gsd')
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
active.active_force['A'] = (1e-15,0,0)
active.active_torque['A'] = (0,0,0)
integrator.forces.append(active)
rotational_diffusion_updater = active.create_diffusion_updater(trigger=1, rotational_diffusion=100.0)
sim.operations += rotational_diffusion_updater

#Thermodynamic properties
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
sim.run(0)

#GSD
gsd_writer = hoomd.write.GSD(filename='trajectory_equil.gsd',trigger=hoomd.trigger.Periodic(10000),mode='wb')
gsd_writer_restart = hoomd.write.GSD(filename='restart.gsd',trigger=hoomd.trigger.Periodic(1000),mode='wb',truncate=True)
sim.operations.writers.append(gsd_writer)
sim.operations.writers.append(gsd_writer_restart)

#Logger
logger = hoomd.logging.Logger()
logger.add(thermodynamic_properties, quantities=['kinetic_energy','potential_energy','pressure','volume'])
logger.add(sim, quantities=['timestep','tps'])
gsd_writer_log = hoomd.write.GSD(filename='log_equil.gsd',trigger=hoomd.trigger.Periodic(100),mode='wb',filter=hoomd.filter.Null(),log=logger)
sim.operations.writers.append(gsd_writer_log)

#Randomize
sim.run(10000)

#Compress
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=20000)
initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_rho = 0.4
final_box.volume = sim.state.N_particles / final_rho
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,box2=final_box,variant=ramp,trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
sim.run(20000)
sim.operations.updaters.remove(box_resize)

#Equilibriate for long time
sim.run(5e5)
