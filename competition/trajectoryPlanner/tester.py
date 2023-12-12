from trajectoryPlanner import TrajectoryPlanner

import scipy

import numpy as np

import matplotlib.pyplot as plt

# GATES = [ [ 0.2 ,  -2.5 ,   1.   ],
#  [ 0.8 ,  -2.5 ,   1.   ],
#  [ 2.  ,  -1.8 ,   0.525],
#  [ 2.  ,  -1.2 ,   0.525],
#  [ 0.3 ,  -0.1 ,   0.525],
#  [-0.3 ,  -0.1 ,   0.525],
#  [-0.4 ,   1.2 ,   1.   ],
#  [-0.4 ,   1.8 ,   1.   ]]

GATES = [[ 0.5,   -2.5  ,  1.   ],
 [ 2. , -1.5  ,  0.525],
 [ 0. ,  0.2  ,  0.525],
 [-0.5,  1.5  ,  1.   ]]
    


OBSTACLES = [  
      [1.5, -2.5, 0, 0, 0, 0],
      [0.5, -1, 0, 0, 0, 0],
      [1.5, 0, 0, 0, 0, 0],
      [-1, 0, 0, 0, 0, 0]
    ]


X0 = [ -0.9, -2.9,  1]

GOAL= [-0.5, 2.9, 0.75]


if __name__ == "__main__": 

  trajPlan = TrajectoryPlanner(X0, GOAL, GATES, OBSTACLES)
  res = trajPlan.optimizer()
  # print(trajPlan.t)
  # trajPlan.plot()

  forcesDirs = trajPlan.sampleForce()

  matrices = trajPlan.sampleOrientations(forcesDirs)

  R_dots = trajPlan.differentiateMatrices(matrices)

  omegas = trajPlan.getOmegas(matrices, R_dots)

  trajPlan.interpolateOmegas(omegas)

  trajPlan.plot_omega()



