import math
import time
import carla
import numpy as np

actor_list = []


def pure_pursuit(tar_location, v_transform):
    L = 2.875
    k = 1
    yaw = v_transform.rotation.yaw * (math.pi / 180)
    x = v_transform.location.x - L / 2 * math.cos(yaw)
    y = v_transform.location.y - L / 2 * math.sin(yaw)
    dx = tar_location.x - x
    dy = tar_location.y - y
    ld = k * math.sqrt(dx ** 2 + dy ** 2)
    alpha = math.atan2(dy, dx) - yaw
    delta = math.atan(2 * math.sin(alpha) * L / ld) * 180 / math.pi
    steer = delta / 45
    if steer > 1:
        steer = 1
    elif steer < -1:
        steer = -1
    return steer


def pid_control(vehicle, target_speed, prev_error):
    Kp = 2.0
    Ki = 0.0
    Kd = 0.5
    target_speed = target_speed
    integral = 0
    dt = 0.1
    total_velocity = total_speed(vehicle)
    error = target_speed - total_velocity
    integral += error * dt
    derivative = (error - prev_error) / dt
    throttle = Kp * error + Ki * integral + Kd * derivative
    brake = 0
    if throttle > 1:
        throttle = 1
    if throttle < 0:
        brake = -throttle
        throttle = 0
    if brake > 1:
        brake = 1
    return throttle, brake, error


def total_speed(vehicle):
    vehicle_v = vehicle.get_velocity()
    vx = vehicle_v.x  # 车辆的x方向线速度
    vy = vehicle_v.y  # 车辆的y方向线速度
    vz = vehicle_v.z  # 车辆的z方向线速度
    total_v = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return total_v


def tracking(vehicle, target_speed):
    waypoint01 = map.get_waypoint(vehicle.get_location(), project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
    v_trans = vehicle.get_transform()
    waypoints = waypoint01.next(10.0)
    waypoint02 = waypoints[0]
    tar_loc = waypoint02.transform.location
    total = total_speed(vehicle)
    v_error = target_speed - total
    steer = pure_pursuit(tar_loc, v_trans)
    throttle, brake, v_error = pid_control(vehicle, target_speed, v_error)
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))


# 递归的方式实现贝塞尔曲线
def bezier(Ps, n, t):
    """递归的方式实现贝塞尔曲线

    Args:
        Ps (_type_): 控制点，格式为numpy数组：array([[x1,y1],[x2,y2],...,[xn,yn]])
        n (_type_): n个控制点，即Ps的第一维度
        t (_type_): 步长t

    Returns:
        _type_: 当前t时刻的贝塞尔点
    """
    if n == 1:
        return Ps[0]
    return (1 - t) * bezier(Ps[0:n - 1], n - 1, t) + t * bezier(Ps[1:n], n - 1, t)


def get_path(vehicle, tar_loc):
    v_loc = vehicle.get_location()
    vx1 = v_loc.x
    vy1 = v_loc.y
    vx2 = tar_loc.x
    vy2 = tar_loc.y
    bias_x = 0.2 * math.cos(yaw)
    bias_y = 0.2 * math.sin(yaw)
    ref = np.array([
        [vx1, vy1],
        [vx1 + bias_x, vy1 + bias_y],
        [vx2, vy2],
        [vx2 + bias_x, vy2 + bias_y]
    ])
    path = []  # 路径点存储
    # 贝塞尔曲线生成
    for t in np.arange(0, 1.01, 0.01):
        p_t = bezier(ref, len(ref), t)
        path.append(p_t)
    path = np.array(path)
    path.tolist()
    paths = []
    for i in range(len(path)):
        paths.append(carla.Location(x=path[i][0], y=path[i][1], z=2.0))
    world.debug.draw_line(paths[0], paths[1], thickness=0.1,
                          life_time=20, color=carla.Color(b=255))
    return paths


def distance(ego, vehicle):
    yaw = ego.get_transform().rotation.yaw
    v1_loc = ego.get_location()
    v2_loc = vehicle.get_location()
    dis_x = (v1_loc.x - v2_loc.x) * math.cos(yaw)
    dis_y = (v1_loc.y - v2_loc.y) * math.sin(yaw)
    # dis = math.sqrt((v1_loc.x * math.cos(v1_yaw) - v2_loc.x * math.cos(v2_yaw)) ** 2
    #                 + (v1_loc.y * math.sin(v1_yaw) - v2_loc.y * math.sin(v2_yaw)) ** 2)
    # dis = dis_x + dis_y
    dis = math.sqrt((v1_loc.x - v2_loc.x) ** 2 + (v1_loc.y - v2_loc.y) ** 2)
    print("-----------------------")
    print(dis_y)
    print(dis_x)
    print(dis)
    return dis_x, dis_y, dis


try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(8.0)

    world = client.get_world()
    map = world.get_map()
    current_map = world.get_map().name

    # 如果当前地图不是'Town04'，则切换到'Town04'
    if current_map != 'Carla/Maps/Town04':
        print("Switching to Town04...")
        try:
            # 加载'Town04'
            world = client.load_world('Town04')
            map = world.get_map()
            print("Switched to Town04.")
        except Exception as e:
            print(f"Error switching to Town04: {e}")

    blueprint_library = world.get_blueprint_library()
    weather = world.get_weather()
    world.set_weather(weather.ClearNight)

    x, y, z = float(-400.0), float(12.8), float(1.0)
    pitch, yaw, roll = float(0.0), float(180.0), float(0.0)
    spawn_point = carla.Transform(location=carla.Location(x, y, z), rotation=carla.Rotation(pitch, yaw, roll))
    v_bp = blueprint_library.filter("bmw")[0]
    vehicle1 = world.spawn_actor(v_bp, spawn_point)
    actor_list.append(vehicle1)
    vehicle1.set_light_state(carla.VehicleLightState.LowBeam)
    vehicle1.set_autopilot(False)

    # 获取主车辆的朝向
    vehicle_transform = vehicle1.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation

    # 计算生成车辆的位置
    distance_behind = 10.0  # 距离主车辆后方的距离

    spawn_point_vehicle = carla.Transform(
        location=carla.Location(
            x=vehicle_location.x - distance_behind * math.cos(math.radians(vehicle_rotation.yaw)),
            y=vehicle_location.y - distance_behind * math.sin(math.radians(vehicle_rotation.yaw)),
            z=vehicle_location.z + 1.0
        ),
        rotation=vehicle_rotation  # 使用主车辆的朝向
    )

    # 生成新车辆
    ev_bp = blueprint_library.filter("model3")[0]
    vehicle = world.spawn_actor(ev_bp, spawn_point_vehicle)
    actor_list.append(vehicle)
    vehicle.set_light_state(carla.VehicleLightState.HighBeam)

    # 生成监视器
    spectator = world.get_spectator()
    transform = vehicle1.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))
    target_speed_v1 = 40 / 3.6

    start_time = time.time()
    while time.time() - start_time <= 2:
        tracking(vehicle1, target_speed_v1)

    # 跟随
    T = 20
    n = 20
    t_s = 0.02
    total_v = total_speed(vehicle)
    target_speed_1 = 50 / 3.6
    v_error = target_speed_1 - total_v
    total_error = 0
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        tracking(vehicle, target_speed_1)
        total_v = total_speed(vehicle)
        total_v1 = total_speed(vehicle1)
        location_vehicle = vehicle.get_location()
        world.debug.draw_line(location_vehicle, location_vehicle, thickness=0.1, life_time=20, color=carla.Color(r=255))
        if dis < 12 and total_v > total_v1:
            break
        time.sleep(t_s)
        print(1)

    # 变道
    total_v = total_speed(vehicle)
    target_speed_2 = 55 / 3.6
    v_error = target_speed_2 - total_v
    start_t = time.time()
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        location_vehicle1 = vehicle1.get_location()
        location_vehicle = vehicle.get_location()
        v_trans = vehicle.get_transform()
        waypoint = map.get_waypoint(location_vehicle1, project_to_road=True, lane_type=carla.LaneType.Driving)
        # 获取左侧车道的车道标记
        left_lane_waypoint = waypoint.next(1.0)[0].get_left_lane()
        # 获取左侧车道的位置
        left_lane_location = left_lane_waypoint.transform.location
        tar_loc = left_lane_location
        # path = get_path(vehicle, tar_loc)
        # tar_loc = path[n]
        steer = pure_pursuit(tar_loc, v_trans)
        throttle, brake, v_error = pid_control(vehicle, target_speed_2, v_error)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        world.debug.draw_line(location_vehicle, location_vehicle, thickness=0.1, life_time=20, color=carla.Color(r=255))
        if dis < 4:
            break
        time.sleep(t_s)
        print(2)

    # 持续超越
    total_v = total_speed(vehicle)
    target_speed_3 = 60 / 3.6
    v_error = target_speed_3 - total_v
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        location_vehicle1 = vehicle1.get_location()
        location_vehicle = vehicle.get_location()
        v_trans = vehicle.get_transform()
        waypoint = map.get_waypoint(location_vehicle1, project_to_road=True, lane_type=carla.LaneType.Driving)
        # 获取左侧车道的车道标记
        left_lane_waypoint = waypoint.next(12.0)[0].get_left_lane()
        # 获取左侧车道的位置
        left_lane_location = left_lane_waypoint.transform.location
        tar_loc = left_lane_location
        # path = get_path(vehicle, tar_loc)
        # tar_loc = path[n]
        steer = pure_pursuit(tar_loc, v_trans)
        throttle, brake, v_error = pid_control(vehicle, target_speed_3, v_error)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        world.debug.draw_line(location_vehicle, location_vehicle, thickness=0.1, life_time=20, color=carla.Color(r=255))
        if dis > 10:
            break
        time.sleep(t_s)
        print(3)

    # 变回原道
    total_v = total_speed(vehicle)
    target_speed_4 = 60 / 3.6
    v_error = target_speed_4 - total_v
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        location_vehicle1 = vehicle1.get_location()
        location_vehicle = vehicle.get_location()
        v_trans = vehicle.get_transform()
        waypoint = map.get_waypoint(location_vehicle1, project_to_road=True, lane_type=carla.LaneType.Driving)
        waypoints = waypoint.next(8 + dis)[0]
        tar_loc = waypoints.transform.location
        steer = pure_pursuit(tar_loc, v_trans)
        throttle, brake, v_error = pid_control(vehicle, target_speed_4, v_error)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        world.debug.draw_line(location_vehicle, location_vehicle, thickness=0.1, life_time=20, color=carla.Color(r=255))
        if dis > 16:
            break
        time.sleep(t_s)

    # 继续行驶
    time_spend = time.time() - start_time
    print("Done overtaking!")
    print("Running time: ", time_spend)
    start_time = time.time()
    while time.time() - start_time <= 2:
        tracking(vehicle1, target_speed_v1)
        tracking(vehicle, target_speed_4)
    for actor in actor_list:
        actor.destroy()
        actor_list.remove(actor)

finally:
    if len(actor_list) == 0:
        print("Finish!")
    else:
        for actor in actor_list:
            actor.destroy()
        print("Finish!")
