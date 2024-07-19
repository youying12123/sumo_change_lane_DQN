import traci
from sumolib import checkBinary
import random
import sys
import random

def setup_simulation(gui=True):
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    traci.start(["sumo-gui", "-c", r"D:\Codes\sumo_study\CMU_DenseTraffic\networks\highway\highway.sumocfg"])

    numVehicles = 20  # 定义车辆总数
    ego_car_id = "ego_car"
    # 主车从0车道到2车道

    traci.vehicle.add(ego_car_id, routeID='route_0', typeID='rl', departLane='0', departPos='0', arrivalLane='2')
    # traci.vehicle.add(ego_car_id, routeID='route_0', typeID='rl', departLane='0', arrivalLane='2')

    traci.vehicle.setLaneChangeMode(ego_car_id, 0)  # 禁用自动变道和防撞功能

    random.seed(2345)

    leader_car_name_list = []
    for i in range(3):
        leader_car_name = 'vehicle_' + str(i)
        traci.vehicle.add(leader_car_name, routeID='route_0', typeID='human', departLane=str(i), departPos='38')
        # traci.vehicle.add(leader_car_name, routeID='route_0', typeID='human', departLane=str(i))

        traci.vehicle.setColor(leader_car_name, (0, 0, 255))
        traci.vehicle.setLaneChangeMode(leader_car_name, 256)  # 禁用自动变道,保证不会发生碰撞
        leader_car_name_list.append(leader_car_name)


    for i in range(3, numVehicles + 3):
        veh_name = 'vehicle_' + str(i)
        # traci.vehicle.add(veh_name, routeID='route_0', typeID='human', departLane='random', departPos='free')
        traci.vehicle.add(veh_name, routeID='route_0', typeID='human', departLane='random', departPos=str(random.uniform(0,40)))

        # traci.vehicle.add(veh_name, routeID='route_0', typeID='human', departLane='random')

        # 以p=0.04换道
        random_i = random.random()
        if random_i <= 0.05:
            lane_change_model = 1621 # 随意换道，但要保证不会发生碰撞
            traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)

        else:
            lane_change_model = 256  # 禁用自动变道,保证不会发生碰撞
            traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)

        # if i == 13:
        #     # traci.vehicle.add(ego_car_id, routeID='route_0', typeID='rl', departLane='0', departPos='0',
        #     #                   arrivalLane='2')
        #     traci.vehicle.add(ego_car_id, routeID='route_0', typeID='rl', departLane='0',
        #                       arrivalLane='2')
        #     traci.vehicle.setLaneChangeMode(ego_car_id, 0)  # 禁用自动变道和防撞功能

        # # 随意换道，除非与 TraCI 冲突，保证不会碰撞
        # lane_change_model = 1621 # 随意换道，除非与 TraCI 冲突，保证不会碰撞
        # traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)

    # 将车辆密集地放置在起点
    for step in range(10):  # 减少初始运行步骤，快速形成密集交通
        traci.simulationStep()

    # 开始循环仿真，实现走走停停行为
    cycle_duration = 20  # 定义周期长度，单位为仿真步骤
    slowdown_factor = 0.7  # 减速因子，用于调整车速至正常速度的一部分
    decel = 1.0
    min_speed = 0.5  # 定义最低速度，单位为米/秒

    last_lane_change_time = {ego_car_id: -float('inf')}
    lane_change_cooldown = 2  # 设定换道冷却时间为2仿真步

    while traci.simulation.getMinExpectedNumber() > 0:
    # for step in range(numVehicles*2):
        current_time = traci.simulation.getTime()

        # 如果主车在车道0或车道1，尝试向左变道
        current_lane = traci.vehicle.getLaneIndex(ego_car_id)
        if current_lane < 2:
            target_lane = current_lane + 1
            if (current_time - last_lane_change_time[ego_car_id] > lane_change_cooldown and
                    traci.vehicle.couldChangeLane(ego_car_id, target_lane)):
                traci.vehicle.changeLane(ego_car_id, target_lane, duration=5)   # 使主车在10仿真步的时间内变到左边的车道
                last_lane_change_time[ego_car_id] = current_time

        for veh_id in traci.vehicle.getIDList():
            if veh_id in leader_car_name_list:  # 只调整最前面三辆车的速度
                if int(current_time) % cycle_duration < cycle_duration / 3:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    # new_speed = max(min_speed, current_speed * slowdown_factor)  # 逐渐减速，但不低于最低速度
                    new_speed = max(min_speed, current_speed - decel * 0.5)  # 逐渐减速，但不低于最低速度

                    traci.vehicle.setSpeed(veh_id, new_speed)
                    # traci.vehicle.setSpeed(veh_id, 0)
                else:
                # elif int(current_time) % cycle_duration >= cycle_duration / 3 and int(current_time) % cycle_duration < cycle_duration * 2 / 3:
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    max_speed = traci.vehicle.getMaxSpeed(veh_id)
                    # new_speed = min(max_speed, current_speed / slowdown_factor)
                    new_speed = min(max_speed, current_speed + decel * 0.5)
                    traci.vehicle.setSpeed(veh_id, new_speed)
                    # print("new_speed: ", new_speed)
                    # print("-------------------------")
                # else:
                #     max_speed = traci.vehicle.getMaxSpeed(veh_id)
                #     traci.vehicle.setSpeed(veh_id,  max_speed)

            else:
                traci.vehicle.setSpeed(veh_id, -1)  # 其他车辆由IDM控制

        traci.simulationStep()
        print("ego position: ", traci.vehicle.getPosition(ego_car_id))
        print("ego angle: ", traci.vehicle.getAngle(ego_car_id))

    traci.close()

if __name__ == "__main__":
    setup_simulation(gui=True)
