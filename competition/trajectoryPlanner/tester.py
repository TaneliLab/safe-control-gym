from trajectoryPlanner import TrajectoryPlanner

import numpy as np

import matplotlib.pyplot as plt

GATES = [  
      [0.5, -2.5, 0, 0, 0, -1.57, 0],
      [2, -1.5, 0, 0, 0, 0, 1],
      [0, 0.2, 0, 0, 0, 1.57, 1],
      [-0.5, 1.5, 0, 0, 0, 0, 0]
    ]


OBSTACLES = [  
      [1.5, -2.5, 0, 0, 0, 0],
      [0.5, -1, 0, 0, 0, 0],
      [1.5, 0, 0, 0, 0, 0],
      [-1, 0, 0, 0, 0, 0]
    ]


X0 = [ -0.9, -2.9,  0.03]

GOAL= [-0.5, 2.9, 0.75]


if __name__ == "__main__": 

    trajPlan = TrajectoryPlanner(X0, GOAL, GATES, OBSTACLES)


    trajPlan.obstacleCost(trajPlan.x)
    


    # t_test = np.linspace(0, 1, 50)

    # res = trajPlan.optimizer()

    # print(res.x)

    
    # plt.plot(t_test, bsplines(t_test))
    # plt.show()