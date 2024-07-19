import time

import gym
import traci
import sumolib
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
import os
import sys
from sumolib import checkBinary
import random

from stable_baselines3 import PPO
from stable_baselines3 import DQN
import torch
from gym import spaces

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def angle_between(p1, p2, rl_angle):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    angle = degrees(atan2(yDiff, xDiff))

    angle += rl_angle
    angle = angle % 360
    return angle


def get_distance(a, b):
    return distance.euclidean(a, b)


class SumoEnv(gym.Env):
    def __init__(self, show_gui=False):
        self.name = 'rlagent'  # egocar
        self.step_length = 0.5
        self.acc_history = deque([0, 0], maxlen=2)
        self.curr_sublane_history = deque([0]*10, maxlen=10)
        self.num_action = 3
        self.grid_state_dim = 3
        self.state_dim = (4 * self.grid_state_dim * self.grid_state_dim) + 1  # 5 info for the agent, 4 for everybody else
        self.pos = (0, 0)
        self.curr_lane = ''
        self.curr_sublane = -1
        self.target_speed = 0
        self.speed = 0
        self.lat_speed = 0
        self.acc = 0
        self.numVehicles = 20  # to change  密集交通设置的他车数量
        self.vType = 'human'  # 车辆类型，他车为'human'，主车为'rl'
        self.lane_ids = []
        self.leader_car_name_list = []
        self.max_steps = 10000  # 单个episode的最大步数
        self.curr_step = 0  # 当前step数
        self.collision = False  # 碰撞标志
        self.done = False  # 结束标志
        self.network_conf = r"D:\Codes\sumo_study\CMU_DenseTraffic\networks\highway\highway.sumocfg"
        self.net = sumolib.net.readNet(r'D:\Codes\sumo_study\CMU_DenseTraffic\networks\highway\highway.net.xml')

        self.t = 0  # 距离目标车辆中心线的横向距离
        self.traget_centerline = -1.6  #目标车道中心线y坐标
        self.jerk = 0

        self.angle = 0
        self.angle_list = deque([0, 0], maxlen=2)
        self.d_angle = 0

        self.deadend_pos = 500.0
        self.start_pos = 0.0

        # ---------------------reward 参数
        self.v_cost = 2.5
        self.t_cost = 10.0  # Lane lateral displacement cost
        self.phi_cost = 1.0  # Lane heading deviation cost
        # self.deadend_cost = 5.0
        self.deadend_cost = 1.25

        self.dis2start_cost = 1.0
        # self.j_cost = 0.001
        self.j_cost = 1.0

        self.ddot_cost = 0.01
        self.a_cost = 0.1  # Acceleration cost
        self.v_des = 5.0

        #--------------- 需要修改
        self.show_gui = show_gui  # 是否开启GUI。建议train过程关闭，test过程打开

        self.action_space = spaces.Discrete(self.num_action)
        high = np.array([600] * self.state_dim)
        low = np.array([-600] * self.state_dim)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        try:
            traci.close()
        except Exception as e:
            print("Error while closing TraCI: ", e)
        self.curr_step = 0
        self.collision = False
        self.done = False

        if self.show_gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
        traci.start(sumoCmd)
        # print("traci.start!!!!")

        # 添加主车,从0车道到2车道
        traci.vehicle.add(self.name, routeID='route_0', typeID='rl', departLane='0', departPos='0', arrivalLane='2')
        # traci.vehicle.add(ego_car_id, routeID='route_0', typeID='rl', departLane='0', arrivalLane='2')

        traci.vehicle.setLaneChangeMode(self.name, 0)  # 禁用自动变道和防撞功能

        random.seed(2345)


        # 三条车道的leader
        self.leader_car_name_list = []
        for i in range(3):
            leader_car_name = 'vehicle_' + str(i)
            traci.vehicle.add(leader_car_name, routeID='route_0', typeID='human', departLane=str(i), departPos='38')
            traci.vehicle.setColor(leader_car_name, (0, 0, 255))
            traci.vehicle.setLaneChangeMode(leader_car_name, 256)  # 禁用自动变道,保证不会发生碰撞
            self.leader_car_name_list.append(leader_car_name)

        # 模拟密集交通,添加他车
        for i in range(3, self.numVehicles + 3):
            veh_name = 'vehicle_' + str(i)
            traci.vehicle.add(veh_name, routeID='route_0', typeID='human', departLane='random',
                              departPos=str(random.uniform(0, 40)))
            # 以p=0.1换道
            if random.random() < 0.1:
                lane_change_model = 1621  # 随意换道，但要保证不会发生碰撞
                traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)
            else:
                lane_change_model = 256  # 禁用自动变道,保证不会发生碰撞
                traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)

        # 制造出dense traffic
        for step in range(20):  # 减少初始运行步骤，快速形成密集交通
            traci.simulationStep()

        self.lane_ids = traci.lane.getIDList()

        self.update_params()
        return self.get_state()


    def update_params(self):

        self.pos = traci.vehicle.getPosition(self.name)
        self.curr_lane = traci.vehicle.getLaneID(self.name)
        self.curr_sublane = int(self.curr_lane.split("_")[1])  # 子路段，比如'gneE6_2' -> 2;
        self.target_speed = traci.vehicle.getMaxSpeed(self.name)
        self.speed = traci.vehicle.getSpeed(self.name)

        self.lat_speed = traci.vehicle.getLateralSpeed(self.name)
        self.acc = traci.vehicle.getAcceleration(self.name)
        self.acc_history.append(self.acc)  # 长度为2的deque，用于后面计算jerk
        self.curr_sublane_history.append(self.curr_sublane)
        # print("curr_sublane_history: ", self.curr_sublane_history)
        self.angle = traci.vehicle.getAngle(self.name)  # 北向为0°，顺时针角度增加
        self.angle_list.append(self.angle)

    def get_state(self):
        state = np.zeros(self.state_dim)
        before = 0
        grid_state = self.get_grid_state().flatten()
        for num, vehicle in enumerate(grid_state):
            if vehicle == 0:
                continue
            if vehicle == -1:  # 主车
                vehicle_name = self.name
                before = 1
            else:
                vehicle_name = 'vehicle_' + (str(int(vehicle)))
            veh_info = self.get_vehicle_info(vehicle_name) # 主车：纵向位置、横向位置、速度、横向速度、加速度
                                                           # 他车：与主车距离(分横纵向)、速度、加速度、横向位置
            idx_init = num * 4  # num：0~8                 # reward修改：去掉动作的reward、修改ttc reward为连续的、
            if before and vehicle != -1:
                idx_init += 1
            idx_fin = idx_init + veh_info.shape[0]
            state[idx_init:idx_fin] = veh_info
        state = np.squeeze(state)  # 37 = 4*8+5  4是他车的4个状态，8是除主车外的8个格子，5是主车的五个状态
        return state

    def get_vehicle_info(self, vehicle_name):
        if vehicle_name == self.name:
            return np.array([self.pos[0], self.pos[1], self.speed, self.lat_speed, self.acc])
        else:
            lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
            long_speed = traci.vehicle.getSpeed(vehicle_name)
            acc = traci.vehicle.getAcceleration(vehicle_name)
            dist = get_distance(self.pos, (lat_pos, long_pos))
            return np.array([dist, long_speed, acc, lat_pos])

    def get_grid_state(self,threshold_distance=10):
        # 九宫格中每个栅格的值：没有车就是0，有主车就是-1，有他车就是他车ID
        agent_lane = self.curr_lane
        agent_pos = self.pos
        edge = self.curr_lane.split("_")[0]  # 主车所在的edge，比如'gneE6_2' -> gneE6
        agent_lane_index = self.curr_sublane
        lanes = [lane for lane in self.lane_ids]

        state = np.zeros([self.grid_state_dim, self.grid_state_dim]) # 九宫格状态
        agent_x = 1
        agent_y = agent_lane_index
        state[agent_x, agent_y] = -1  # 主车所在位置为-1
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            veh_lane = int(lane.split("_")[1])
            for vehicle in vehicles:
                if vehicle == self.name:
                    continue
                veh_pos = traci.vehicle.getPosition(vehicle)
                if get_distance(agent_pos, veh_pos) > threshold_distance:
                    continue

                # 获取九宫格内相对主车的位置
                rl_angle = traci.vehicle.getAngle(self.name)  # 大部分是90°，sumo默认正北为0度顺时针转
                angle = angle_between(agent_pos, veh_pos, rl_angle)

                veh_id = vehicle.split("_")[1]
                # 以车辆朝向为北
                if angle > 337.5 or angle < 22.5:         # 东
                    state[agent_x, veh_lane] = veh_id
                if angle >= 22.5 and angle < 67.5:        # 东北
                    state[agent_x - 1, veh_lane] = veh_id
                if angle >= 67.5 and angle < 112.5:       # 北
                    state[agent_x - 1, veh_lane] = veh_id
                if angle >= 112.5 and angle < 157.5:      # 西北
                    state[agent_x - 1, veh_lane] = veh_id
                if angle >= 157.5 and angle < 202.5:      # 西
                    state[agent_x, veh_lane] = veh_id
                if angle >= 202.5 and angle < 237.5:      # 西南
                    state[agent_x + 1, veh_lane] = veh_id
                if angle >= 237.5 and angle < 292.5:      # 南
                    state[agent_x + 1, veh_lane] = veh_id
                if angle >= 292.5 and angle < 337.5:       # 东南
                    state[agent_x + 1, veh_lane] = veh_id

        state = np.fliplr(state)
        return state

    def adjust_speeds(self):
        # 实现走走停停行为
        cycle_duration = 20  # 定义周期长度，单位为仿真步骤
        # slowdown_factor = 0.7  # 减速因子,用于调整车速至正常速度的一部分
        decel = 1.0 # 加/减速度
        min_speed = 0.5  # 定义最低速度，m/s

        current_time = traci.simulation.getTime()
        for veh_id in traci.vehicle.getIDList():
            if veh_id in self.leader_car_name_list:
                if int(current_time) % cycle_duration < cycle_duration / 3:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    # new_speed = max(min_speed, current_speed * slowdown_factor)
                    new_speed = max(min_speed, current_speed - decel * 0.5)  # 逐渐减速，但不低于最低速度
                    traci.vehicle.setSpeed(veh_id, new_speed)
                else:
                    # elif int(current_time) % cycle_duration >= cycle_duration / 3 and int(current_time) % cycle_duration < cycle_duration * 2 / 3:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    max_speed = traci.vehicle.getMaxSpeed(veh_id)
                    # new_speed = min(max_speed, current_speed / slowdown_factor)
                    new_speed = min(max_speed, current_speed + decel * 0.5)
                    traci.vehicle.setSpeed(veh_id, new_speed)
            else:
                traci.vehicle.setSpeed(veh_id, -1)  # 其他车辆由IDM控制


    def step(self, action):
        # 流程：执行换道->调速->模拟器更新一步->计算碰撞->更新参数->计算reward(与collision与更新的状态有关)->计算done、获取next state
        if action != 0:
            if action == 1:  # 右变道，当前处于1车道，则会到达0车道，限制5个时间步内完成变道
                             #       当前处于2车道，则会到达1车道,限制5个时间步内完成变道
                if self.curr_sublane == 1:
                    traci.vehicle.changeLane(self.name, 0, duration=2)
                elif self.curr_sublane == 2:
                    traci.vehicle.changeLane(self.name, 1, duration=2)
            if action == 2:  # 左变道，当前处于0车道，则会到达1车道
                             #       当前处于1车道，则会到达2车道，限制2个时间步内完成变道
                if self.curr_sublane == 0:
                    traci.vehicle.changeLane(self.name, 1, duration=2)
                elif self.curr_sublane == 1:
                    traci.vehicle.changeLane(self.name, 2, duration=2)

        # 调整车速以模拟走走停停行为
        self.adjust_speeds()
        traci.simulationStep()
        collision = self.detect_collision()
        self.update_params()
        closest_dis, safe_dis, closest_vehicle_id = self.getDisEgo2ClosestVehicle()
        reward = self.compute_reward(collision, action, closest_dis, safe_dis)
        next_state = self.get_state()
        done = self.getDoneState(self.curr_sublane)
        self.curr_step += 1
        info = {"closest_dis": closest_dis, "safe_dis": safe_dis, "action": action, "id": closest_vehicle_id}

        if done:
            traci.close()

        return next_state, reward, done, info

    def getDisEgo2ClosestVehicle(self):
        closest_dis = float('inf')
        vehicles = traci.lane.getLastStepVehicleIDs(self.curr_lane)
        for vehicle in vehicles:
            if vehicle == self.name:
                continue
            veh_pos = traci.vehicle.getPosition(vehicle)
            disEgo2ClosestVehicle = get_distance(self.pos, veh_pos)
            if disEgo2ClosestVehicle < closest_dis:
                closest_dis = disEgo2ClosestVehicle
                closest_vehicle_id = vehicle
                closest_vehicle_speed = traci.vehicle.getSpeed(closest_vehicle_id)
                # 两秒规则
                safe_dis = closest_vehicle_speed * 1.75
        return closest_dis, safe_dis, closest_vehicle_id


    def getDoneState(self, ego_lane):
        done = False
        if self.collision:
            done = True
            print("Collision occurs!")
            print("*****************************************")
            return done

        # 成功条件应该是保持在2车道一段时间就结束(连续10个时间步)，并且没有快达到deadend
        if ego_lane == 2 and all(x == 2 for x in self.curr_sublane_history) and self.pos[0] <= (self.deadend_pos - 20.0) and self.angle == 90.0:
            done = True
            print("Mission success!")
            print("*****************************************")
            return done

        # 失败条件应该是快达到deadend了
        if self.pos[0] > self.deadend_pos - 20.0:
            done = True
            print("Mission failure!")
            print("*****************************************")
            return done

        return done

    def detect_collision(self):
        collisions = traci.simulation.getCollidingVehiclesIDList()
        if self.name in collisions:
            self.collision = True
            return True
        return False

    def compute_reward(self, collision, action, closest_dis, safe_dis):

        # --------------------
        # SZ:
        # 1. 完成(子)任务的奖励
        R_inlane = 0.0
        if self.curr_sublane == 2:    # 在2车道奖励
            R_inlane = 5.0
            print("我到2车道了")
        elif self.curr_sublane == 1:  # 在1车道奖励
            print("我到1车道了")
            R_inlane = 1.5
        else:
            print("我在0车道")

        # 2.惩罚右转（主要是避免到达2车道后刷分）
        R_right = 0.0
        if action == 1:
            R_right = -100.0

        # 3. 惩罚时间
        R_time = -1.0

        # 4. 惩罚与deadend的距离
        distance_to_deadend = self.compute_distance_to_deadend()
        R_deadend = -(self.deadend_cost * (500 / distance_to_deadend))

        # 5.惩罚碰撞
        R_collision = 0.0
        if collision:
            R_collision = -200.0

        # # 6.惩罚过于勉强的换道: 换道后距离最近车辆的距离不能太小
        R_close = 0.0
        if action == 2 or action == 1:
            if closest_dis < safe_dis:
                R_close = -200.0

        # R_total = R_inlane + R_action + R_time + R_deadend + R_collision
        R_total = R_inlane + R_time + R_deadend + R_collision + R_close + R_right
        print("R_inlane: {0}, R_right: {1}, R_time: {2}, R_deadend:{3}, R_close:{4}, R_total:{5:.3f}".format(R_inlane, R_right, R_time, R_deadend, R_close, R_total))
        # print("R_inlane: {0}, R_action: {1}, R_time: {2}, R_deadend:{3}, R_total:{4:.3f}".format(R_inlane, R_action, R_time, R_deadend, R_total))

        return R_total

    def compute_lateral_displacement(self):
        return abs(self.traget_centerline - self.pos[1])

    def compute_distance_to_deadend(self):
        return abs(self.deadend_pos - 20.0 - self.pos[0])

    def compute_distance_to_start(self):
        return abs(self.pos[0] - self.start_pos)

    def compute_jerk(self):
        return (self.acc_history[1] - self.acc_history[0]) / self.step_length

    def compute_ddot(self):
        return (self.angle_list[1] - self.angle_list[0]) / 1

    def render(self, mode='human', close=False):
        self.show_gui = True
        pass

    def close(self):
        traci.close(False)


if __name__ == '__main__':
    env = SumoEnv(show_gui=False)
    start_time = time.time()

    # model = PPO('MlpPolicy', env,
    #             policy_kwargs=dict(net_arch=[64, 64]),
    #             learning_rate=5e-4,
    #             batch_size=32,
    #             gamma=0.99,
    #             verbose=1,
    #             tensorboard_log="logs/",
    #             )

    model = DQN('MlpPolicy',
                env,
                learning_rate=5e-4,
                batch_size=64,
                buffer_size=50000,
                learning_starts=0,
                target_update_interval=250,
                policy_kwargs=dict(net_arch=[64, 64]),
                verbose=1,
                tensorboard_log="logs/")

    model.learn(int(1e5))
    model.save("models/dqn18")

    end_time = time.time()
    running_time = end_time - start_time

    print('time cost : %.5f sec' % running_time)






