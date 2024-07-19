import os
import sys
import time
from GymEnv_LC2013 import *

from stable_baselines3 import DQN
import time

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary


# if_show_gui = True
#
# if not if_show_gui:
#     sumoBinary = checkBinary('sumo')
# else:
#     sumoBinary = checkBinary('sumo-gui')

# sumocfgfile = "sumo_config/my_config_file.sumocfg"
# traci.start([sumoBinary, "-c", sumocfgfile])

if __name__ == '__main__':

    env = SumoEnv(show_gui=True)

    # load model
    model = DQN.load("models/dqn17", env=env)

    eposides = 10

    for eq in range(eposides):
        print("Test eposide: {}".format(eq))
        obs = env.reset()
        done = False
        rewards = 0
        counts = 0
        while not done:
            counts += 1
            time.sleep(1)
            action, _state = model.predict(obs, deterministic=True)
            action = action.item()
            obs, reward, done, info = env.step(action)
            print(info)
            # env.render()
            rewards += reward
        print("The rewards is: {},  the counts is: {}".format(rewards, counts))

