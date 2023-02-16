import espressomd
import espressomd.shapes
import espressomd.observables

from espressomd import visualization
import espressomd.visualization

import numpy as np
import matplotlib.pyplot as plt
import json

import espressomd.lb
import espressomd.lbboundaries

import numpy as np
import matplotlib.pyplot as plt
from espressomd.virtual_sites import VirtualSitesOff, VirtualSitesRelative
import pyvoro
from operator import itemgetter
from espressomd.observables import ParticlePositions, ParticleDistances
import espressomd.observables
import espressomd.accumulators
from espressomd.accumulators import Correlator
from espressomd.observables import ParticlePositions, ParticleVelocities
from moviepy.editor import *
from moviepy.editor import ImageSequenceClip



Phi_0 = 0.6 
num_try = 10

num_frame = 100 # gif를 사진 몇장으로 만들지 정함

save_dir = './test_2/Fig/' # jpg 및 gif 저장할 곳

data_dir = './test_2/'
with open(data_dir + "data_5000000_6.json", 'r') as f:
    data = json.load(f)



''' Set parameters '''
d_S = 6 # data['d_S'] # 1 / lattice # diameter of small particle
d_L = 4 #data['d_L'] # diameter of small particle
d_SL = 0.5*(d_S + d_L)

SR = d_L/d_S # size ratio
H_0 = data['H_0'] # 1.6 * d_S # initial height
X = data['X']
Z = data['Z']
# W = data['W'] # 24 * d_S # system width

# m_0 = data['m_0'] # 6 / (np.pi*np.power(d_L, 3)) # unit mass
# m_S = m_0*np.pi*np.power(d_S, 3)/6
# m_L = m_S * np.power(SR, 3) #m_0*np.pi*np.power(d_L, 3)/6

PP = data['particle_pos']
divide = int(len(PP)/num_frame)


print(len(PP))
print(divide)

V_P_S = 1/4 * np.pi * ((d_S)**2) # single small particle surface area
V_P_L = 1/4 * np.pi * ((d_L)**2) # single large particle surface area

N_S = data['N_S']
N_L = 0# data['N_L']
N_T = N_S + N_L
N_I = 2

type_S = 6
type_L = 9
type_wall = 1

GAMMA = 1 # friction coeff. # 아무거나 넣으면 됨
kT = 1 # thermal energy  # 아무거나 넣으면 됨
EPSILON = 1

TIME_STEP = 0.001 # 아무거나 넣으면 됨


''' Set espresso system box '''
system = espressomd.System(box_l=[X, H_0, Z])
# system.set_random_state_PRNG()
# np.random.seed(seed=system.seed)
system.cell_system.skin = 0.4
system.time_step = TIME_STEP
system.periodicity = [True, False, True]

''' Set walls '''
TOP_WALL = system.constraints.add(shape=espressomd.shapes.Wall(
    dist=-H_0, normal=[0, -1, 0]),particle_type=type_wall, penetrable=False)

BOT_WALL = system.constraints.add(shape=espressomd.shapes.Wall(
    dist=0, normal=[0, 1, 0]),particle_type=type_wall, penetrable=False)


''' Set interactions'''
#espressomd.assert_features(['LB_BOUNDARIES'])
#required_features = ["WCA", "LENNARD_JONES", "VIRTUAL_SITES", "VIRTUAL_SITES_RELATIVE", 'LB_BOUNDARIES']

# small-small
system.non_bonded_inter[type_S, type_S].wca.set_params(epsilon=EPSILON, sigma=d_S)

# small-wall
system.non_bonded_inter[type_S, type_wall].wca.set_params(epsilon=EPSILON, sigma=d_S/2)

# large-large
system.non_bonded_inter[type_L, type_L].wca.set_params(epsilon=EPSILON, sigma=d_L)

# large-small
system.non_bonded_inter[type_L, type_S].wca.set_params(epsilon=EPSILON, sigma=d_SL)

# large-wall
system.non_bonded_inter[type_wall, type_L].wca.set_params(epsilon=EPSILON, sigma=d_L/2)





''' Set visualizer '''
visualizer = espressomd.visualization.openGLLive(
    system, window_size=[500, 500], particle_sizes = [1, 1, 1, 1, 1, 1, 0.5],
    background_color=[1, 1, 1],camera_position=[0.5*X, 0.5*H_0, 2.5*Z], drag_enabled=True, drag_force=GAMMA*10)

rod_internal_bond_1 = espressomd.interactions.HarmonicBond(k = 1, r_0 = 8)
system.bonded_inter.add(rod_internal_bond_1)
N_P = 10

for n in range(0, len(PP), divide):
    # print(n)
    for i in range(N_S):
        system.part.add(id=N_P*i+0, pos=[PP[n][0+3*(N_P*i)], PP[n][1+0+3*(N_P*i)], PP[n][2+0+3*(N_P*i)]], type=type_S)
        for p in range(1,N_P):
            # print(p)
            system.part.add(id=N_P*i+p, pos=[PP[n][3*(N_P*i+p)], PP[n][1+3*(N_P*i+p)], PP[n][2+3*(N_P*i+p)]], type=type_S)
        # system.part.by_id(N_P*i+p).add_bond((rod_internal_bond_1, N_P*i))
    for i in range(N_S, (N_S+N_L)):
        system.part.add(id=N_P*i+0, pos=[PP[n][0+3*(N_P*i)], PP[n][1+0+3*(N_P*i)], PP[n][2+0+3*(N_P*i)]], type=type_S)
        for p in range(1,N_P):
            # print(p)
            system.part.add(id=N_P*i+p, pos=[PP[n][3*(N_P*i+p)], PP[n][1+3*(N_P*i+p)], PP[n][2+3*(N_P*i+p)]], type=type_S)
        # system.part.by_id(N_P*i+p).add_bond((rod_internal_bond_1, N_P*i))

    # visualizer.run()
    
    visualizer.screenshot(save_dir + 'test_{:0>5}.png'.format(n))
    system.part.all().remove()
    

'''Image to gif'''
images = [] # load images 
for n in range(0, len(PP), divide):
#for n in range(0, len(PP)):
    images.append(save_dir + 'test_{:0>5}.png'.format(n))

clip = ImageSequenceClip(images,fps=30)
clip.write_gif(save_dir + 'movie.gif')