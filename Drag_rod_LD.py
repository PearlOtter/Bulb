from ast import Eq
from ctypes.wintypes import POINT
from hashlib import sha1
from locale import D_FMT
from math import gamma
import numpy as np
import json
import time
import datetime

import espressomd
import espressomd.shapes
# import espressomd.visualization_opengl
import espressomd.interactions
import espressomd.lb
import espressomd.lbboundaries
from espressomd import visualization
import espressomd.virtual_sites
import espressomd.polymer

RUN_DIR = './test_1/'
num_try = 5
save_dir_part = './Data_part/'

start = time.time()

###### GEOMETRY SETUP #######
L = 10
H = 41 # 2.0*L
X = 40 # 2.0*L
Z = 40 # 2.0*L
AGRID = 1.0
WALL_OFFSET = AGRID
WALL_VELOCITY = 1.0e-1
EXT_FORCE = [0.0, 0.0, 0.0]

###### SYSTEM SETUP #######
TAU = 1
KINEMATIC_VISCOSITY = ((2*TAU)-1)/6
DENSITY = 36.0
VISCOSITY = KINEMATIC_VISCOSITY*DENSITY

kT = 1
Seed = np.random.randint(1,1000)

###### PARTICLE SETUP #######
A = 1.5
n_rod_part = 2 
#### 2 for rigid dumbell
#### A for rigid 
d_R = 1.0
d_S = 1.0

type_S = 1
type_WALL = 0

nL3 = 64
Vol_P_R = 1/6*np.pi*((d_R)**3)
N_S = int(768/2) # int(nL3*((X/L)*((H-2*WALL_OFFSET)/L)*(Z/L)))
m_R = DENSITY * Vol_P_R
# Phi_real = (N_S*n_rod_part*Vol_P_R)/((H-(2*WALL_OFFSET+d_R))*(X)*(Z))
print(N_S)

###### PARTICLE PROPERTIES (LD) ######
GAMMA_R = (1.0)*6*np.pi*VISCOSITY*(d_S)
eps_RR = 48. # Small-Small
sig_RR = d_R

data = {
    'H' : H,
    'X' : X,
    'Z' : Z,
    'Kinematic viscosity' : KINEMATIC_VISCOSITY,
    'Density' : DENSITY,
    'n_col_particle' : n_rod_part
}

with open(RUN_DIR + 'init.txt', 'w') as f:
    json.dump(data, f)

TIME_STEP = 0.001   ### LD TIME STEP ###
TIME_STEP_LB = 0.01   ### LB TIME STEP ###

radius_col_S = 3.0

Time_Scale = (6*np.pi*VISCOSITY*d_R*(L*L))/kT
# print(Time_Scale)
# DR_0 = 3*kT*np.log(L/radius_col_S)/(np.pi*VISCOSITY*L*L*L)
# print(DR_0)
# DT_0 = 3*kT*np.log(L/radius_col_S)/(np.pi*VISCOSITY*L*L*L)
# print(DT_0)

###### START SIMULATION SYSTEM ######
system = espressomd.System(box_l = [X, H, Z])
# system.set_ramdom_state_PRNG()
np.random.seed(seed=Seed)
system.virtual_sites = espressomd.virtual_sites.VirtualSitesRelative()
system.time_step = 0.01*TIME_STEP

EPSILON_FENE = 30*kT
###### SET ROD CONNECTION ######
# rod_internal_bond_1 = espressomd.interactions.HarmonicBond(k = 5, r_0 = A)
# system.bonded_inter.add(rod_internal_bond_1)
pol_internal_bond_1 = espressomd.interactions.FeneBond (k= EPSILON_FENE, d_r_max = 1.5*d_S)
system.bonded_inter.add(pol_internal_bond_1)
pol_internal_bond_2 = espressomd.interactions.RigidBond(r = 1.0*d_S)
system.bonded_inter.add(pol_internal_bond_2)

visualizer = espressomd.visualization.openGLLive(
    system, window_size = [900, 900], 
    background_color = [1, 1, 1],
    camera_position = system.box_l*[0.5, 0.5, 3],
    particle_sizes = [d_S/2, AGRID/2],
    # LB_draw_velocity_plane=True,
    # LB_plane_ngrid=15,
    drag_enabled = True, drag_force = 10
    
) 

TOP_WALL = system.constraints.add(shape=espressomd.shapes.Wall(
    normal=[0, 1, 0], dist=WALL_OFFSET) ,particle_type=type_WALL, penetrable=False) #, particle_velocity = [-WALL_VELOCITY,0,0])

BOT_WALL = system.constraints.add(shape=espressomd.shapes.Wall(
    normal=[0, -1, 0], dist=-(H - WALL_OFFSET)) ,particle_type=type_WALL, penetrable=False) #, particle_velocity = [WALL_VELOCITY,0,0])

######### Purely repulsive WCA potential #########
EPSILON_LJ = kT
# small-small
system.non_bonded_inter[type_S, type_S].wca.set_params(epsilon=EPSILON_LJ, sigma=d_S)

# small-wall
system.non_bonded_inter[type_S, type_WALL].wca.set_params(epsilon=EPSILON_LJ, sigma=d_S/2)

EPSILON_bend = 500*kT
######## Bending potential ########
angle_cosine = espressomd.interactions.AngleCosine(bend=EPSILON_bend, phi0= 180*np.pi/180)
system.bonded_inter.add(angle_cosine)

# harmonic_radius_S = radius_col_S-0.5
n_pol_part = 10

# pos_rasp_S = np.load(save_dir_part + "part_pos_"+str(harmonic_radius_S)+'_'+str(n_col_part)+'.npy')
# gamma_R = 6*np.pi*VISCOSITY*d_S
# # center = [X/2, H*(1/4), Z/2]

# pos_init = np.zeros((N_S, 3))
# for i in range(N_S):
#     pos_init[i,:] = np.random.random(3)*[X, H-(A*n_pol_part+2*WALL_OFFSET), Z] + (1/2)*A*n_pol_part + WALL_OFFSET
# print(pos_init)

positions = espressomd.polymer.linear_polymer_positions(
    n_polymers = N_S, 
    beads_per_chain = n_pol_part,
    bond_length = 1,
    seed = Seed,
    # start_positions = pos_init,
    min_distance = 1.0,
    bond_angle = 180, # *np.pi/180,
    max_tries = 1*1000,
    respect_constraints = True)
    
# print(positions)

for i in range(len(positions)):
    p_previous = None
    for n in range(n_pol_part):
        # print(n)
        pos = positions[i][n]
        # print(pos)
        p = system.part.add(pos=pos, id = i*n_pol_part+n,
                            type = type_S, mass = m_R*(d_S**3))
        if p_previous is not None:
            p.add_bond((pol_internal_bond_1, p_previous))
        p_previous = p
    
for i in range (len(positions)):
    for n in range(1, n_pol_part-1):
        # print(n)
        system.part.by_id(i*n_pol_part+n).add_bond((angle_cosine, i*n_pol_part+n-1, i*n_pol_part+n+1))       

##################################################################################
########################### System Equilibrium ###################################
##################################################################################
system.cell_system.skin = 0.4
# visualizer.run()
test_dir = RUN_DIR + './temp_1/'
visualizer.screenshot(test_dir + 'test_'+'000'+'.png'.format(n))

''' Set equilibrium '''
system.integrator.set_steepest_descent(
    f_max=0, gamma=1, max_displacement=0.02)
system.integrator.run(1000)
system.integrator.set_vv() 

ljcap = 10
CapSteps = 100
for i in range(CapSteps):
    print('CapSteps = '+str(i))
    system.force_cap = ljcap
    system.integrator.run(200)
    ljcap += 5
system.force_cap = 0

system.thermostat.set_langevin(kT = 1.0*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.time_step = TIME_STEP
system.cell_system.skin = 0.4
system.integrator.run(5000)

system.thermostat.set_langevin(kT = 0.1*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.integrator.run(5000)

system.thermostat.set_langevin(kT = 0.01*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.integrator.run(5000)

system.thermostat.set_langevin(kT = 0.0*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.integrator.run(5000)

system.virtual_sites = espressomd.virtual_sites.VirtualSitesRelative()
for i in range (len(positions)):
    for n in range(1, n_pol_part):
        system.part.by_id(i*n_pol_part+n).delete_bond((pol_internal_bond_1, i*n_pol_part+n-1))
        system.part.by_id(i*n_pol_part+n).vs_auto_relate_to(i*n_pol_part)
        
for i in range (len(positions)):
    for n in range(1, n_pol_part-1):
        system.part.by_id(i*n_pol_part+n).delete_bond((angle_cosine, i*n_pol_part+n-1, i*n_pol_part+n+1))     

system.thermostat.set_langevin(kT = 0.1*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.integrator.run(5000)

system.thermostat.set_langevin(kT = 0.0*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.integrator.run(5000)

# visualizer.run()
system.thermostat.turn_off()

# R_g = system.analysis.calc_rg(chain_start = 0, number_of_chains = N_S, chain_length = n_pol_part)
# print('Radius of gyration ='+str(R_g))

##################################################################################
############################# LD simulation ######################################
#################################################################################
print('Set LD system')
system.thermostat.set_langevin(kT = 0.1*kT, gamma = GAMMA_R, 
                               seed = np.random.randint(1, 1000))
system.time_step = TIME_STEP
system.cell_system.skin = 0.4

##################################################################################
##################################################################################
##################################################################################

##################################################################################
############################### LB simulation ####################################
##################################################################################
# print('Add LB fluids')
# # gt.kill_particle_motion()
# # gt.kill_particle_forces()
# # Seed = np.random.randint(1,1000)

# lbf = espressomd.lb.LBFluidGPU(kT = 0.0*kT, seed = Seed,
#                             agrid = AGRID, dens = DENSITY, visc = 1*VISCOSITY,
#                             tau = TIME_STEP_LB, ext_force_density = EXT_FORCE
#                             )
# system.actors.add(lbf)
# system.thermostat.turn_off()
# system.thermostat.set_lb(LB_fluid = lbf, seed = Seed, gamma = GAMMA_R)
# system.time_step = TIME_STEP
# system.cell_system.skin = 0.4
##################################################################################
##################################################################################
##################################################################################

visualizer = espressomd.visualization.openGLLive(
    system, window_size = [900, 900], 
    background_color = [1, 1, 1],
    camera_position = system.box_l*[0.5, 0.5, 3],
    particle_sizes = [d_S/2, AGRID/2],
    drag_enabled = True, drag_force = 10   
) 
# visualizer.run() 
##################################################################################
##################################################################################
##################################################################################

# WALL_OFFSET = AGRID
# TOP_Wall_LB = espressomd.lbboundaries.LBBoundary(
#     shape=espressomd.shapes.Wall(normal=[0, 1, 0], dist=WALL_OFFSET), velocity = [-WALL_VELOCITY,0,0])
# Bottom_Wall_LB = espressomd.lbboundaries.LBBoundary(
#     shape=espressomd.shapes.Wall(normal=[0, -1, 0], dist=-(H - WALL_OFFSET)), velocity = [WALL_VELOCITY,0,0])

# system.lbboundaries.add(TOP_Wall_LB)
# system.lbboundaries.add(Bottom_Wall_LB)

########## Express the time for setting the system #####################
sett = time.time()
sec = (sett-start)
CalTime = datetime.timedelta(seconds=sec)
print(CalTime)
########################################################################

# system.integrator.run(1000)
print('End equilibrium')
# visualizer.run() 

#######################################################################
################# Test simulation setting #############################
#######################################################################
from moviepy.editor import ImageSequenceClip
test_dir = RUN_DIR + './temp_1/'
for i in range(2000):
    visualizer.screenshot(test_dir + 'test_'+str(i)+'.png'.format(n))
    for h in range(100):
        Y = system.part.all().pos[:,1]
        # print(Y)
        # print(Y-(H/2))
        V = system.part.all().v[:,0]
        u_f = (WALL_VELOCITY/(H/2))*(Y-(H/2))
        # print(u_f)
        F = GAMMA_R*(u_f-V)
        for p in range(len(Y)):
            system.part.by_id(p).ext_force = [F[p],0,0]
        # print(system.part.all().ext_force)
        system.integrator.run(1)

'''Image to gif'''
images = [] # load images 
for n in range(1000):
#for n in range(0, len(PP)):
    images.append(test_dir + 'test_'+str(n)+'.png'.format(n))

clip = ImageSequenceClip(images,fps=30)
clip.write_gif(test_dir + 'movie.gif')
print('End test GIF')

# Iterate until the flow profile converges (5000 LB updates)
########## Express calculation procsess and the time used #####################
iter = 0
for s in range(1,1001):
    Particle_Position = []
    Particle_Orientation = []
    for i in range(1, 101): 
        for j in range(1):
            PP = []
            PO = []
            for k in range(N_S):
                for h in range(n_pol_part):
                    PP.append(system.part.by_id(k*n_pol_part+h).pos[0])
                    PP.append(system.part.by_id(k*n_pol_part+h).pos[1])
                    PP.append(system.part.by_id(k*n_pol_part+h).pos[2])
                # PP.append(system.part.by_id(k*n_pol_part).pos[0])
                # PP.append(system.part.by_id(k*n_pol_part).pos[1])
                # PP.append(system.part.by_id(k*n_pol_part).pos[2])
                # PP.append(system.part.by_id(k*n_pol_part+(n_pol_part-1)).pos[0])
                # PP.append(system.part.by_id(k*n_pol_part+(n_pol_part-1)).pos[1])
                # PP.append(system.part.by_id(k*n_pol_part+(n_pol_part-1)).pos[2])
                PO_temp = []
                PO_temp.append((system.part.by_id(k*n_pol_part+(n_pol_part-1)).pos[0]-system.part.by_id(k*n_pol_part).pos[0]))
                PO_temp.append((system.part.by_id(k*n_pol_part+(n_pol_part-1)).pos[1]-system.part.by_id(k*n_pol_part).pos[1]))
                PO_temp.append((system.part.by_id(k*n_pol_part+(n_pol_part-1)).pos[2]-system.part.by_id(k*n_pol_part).pos[2]))
                PO_temp = PO_temp/np.linalg.norm(PO_temp)
                PO.append(PO_temp[0])
                PO.append(PO_temp[1])
                PO.append(PO_temp[2])
            Particle_Position.append(PP)
            Particle_Orientation.append(PO)
            # system.integrator.run(100)
            for h in range(1000):
                Y = system.part.all().pos[:,1]
                V = system.part.all().v[:,0]
                u_f = (WALL_VELOCITY/(H/2))*(Y-(H/2))
                # print(u_f)
                F = GAMMA_R*(u_f-V)
                system.part.all().ext_force[:,0] = F
                system.integrator.run(1)
            iter = iter+1000
            # print(PP)
            # print(PO)

            
            
        print('iteration number = ' + str(iter))
        Itert = time.time()
        sec = (Itert-start)
        CalTime = datetime.timedelta(seconds=sec)
        print(CalTime)
    # fluid_velocities = (lbf[:,:,:].velocity)[:,:,:,0]
    # fluid_velocities = np.average(fluid_velocities, axis=(0,1,2))
    # print(fluid_velocities)
    # Re_p = (DENSITY*fluid_velocities*H/VISCOSITY)*((d_S/H)**2)

    time_plot = np.arange((iter-100000)/100, (iter/100))
    time_plot = time_plot*100*TIME_STEP # /Time_Scale
    data = {
            # "d_L": d_L,
            "H_0": H,
            "X" : X,
            "Z" : Z,
            # "m_0": m_0,
            # "nu_0": nu_0,
            # "eta_0": eta_0,
            "kT": kT,
            "N_S": N_S,
            # "N_L": N_L,
            # "D_R0": DR_0,
            # "D_T0": DT_0,
            # "u_avg": float(fluid_velocities),
            # "particle_Reynolds" : float(Re_p),
            # "D_L0": D_L0,
    #        "tau_B": tau_B,
            "time_plot": time_plot.tolist()
            # 'particle_pos': Particle_Position
                                    }

    data['particle_pos'] = Particle_Position
    data['particle_ori'] = Particle_Orientation

    data_dir = './test_1/'  # 파일을 저장할 곳
    with open(data_dir + "data_"+str(iter)+'_'+str(num_try)+".json", 'w') as f:
        json.dump(data, f)


