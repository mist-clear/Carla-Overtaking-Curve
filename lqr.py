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
    vx = v_loc.x
    vy = v_loc.y
    ref = np.array([
        [vx, vy],
        [tar_loc[0].x, tar_loc[0].y],
        [tar_loc[1].x, tar_loc[1].y],
        [tar_loc[2].x, tar_loc[2].y]
    ])
    path = []  # 路径点存储
    # 贝塞尔曲线生成
    for t in np.arange(0, 1.0, 0.01):
        p_t = bezier(ref, len(ref), t)
        path.append(p_t)
    path = np.array(path)
    path.tolist()
    paths = []
    for i in range(len(path)):
        paths.append(carla.Location(x=path[i][0], y=path[i][1], z=v_loc.z + 0.2))
    for i in range(10, len(path) - 1, 10):
        world.debug.draw_line(paths[i - 10], paths[i], thickness=0.1,
                              life_time=0.1, color=carla.Color(b=255))
    world.debug.draw_line(paths[0], paths[1], thickness=0.1,
                          life_time=25, color=carla.Color(r=255))
    return path


N = 100  # 迭代范围
EPS = 1e-4  # 迭代精度
Q = np.eye(3) * 1
R = np.eye(2) * 20
dt = 0.1  # 时间间隔，单位：s
L = 2.875  # 车辆轴距，单位：m


def MyReferencePath(path):
    # set reference trajectory
    # refer_path包括4维：位置x, 位置y， 轨迹点的切线方向, 曲率k
    refer_path = np.zeros((100, 4))
    refer_path[:, 0] = path[:, 0]  # x
    refer_path[:, 1] = path[:, 1]  # y
    # 使用差分的方式计算路径点的一阶导和二阶导，从而得到切线方向和曲率
    for i in range(len(refer_path)):
        if i == 0:
            dx = refer_path[i + 1, 0] - refer_path[i, 0]
            dy = refer_path[i + 1, 1] - refer_path[i, 1]
            ddx = refer_path[2, 0] + refer_path[0, 0] - 2 * refer_path[1, 0]
            ddy = refer_path[2, 1] + refer_path[0, 1] - 2 * refer_path[1, 1]
        elif i == (len(refer_path) - 1):
            dx = refer_path[i, 0] - refer_path[i - 1, 0]
            dy = refer_path[i, 1] - refer_path[i - 1, 1]
            ddx = refer_path[i, 0] + refer_path[i - 2, 0] - 2 * refer_path[i - 1, 0]
            ddy = refer_path[i, 1] + refer_path[i - 2, 1] - 2 * refer_path[i - 1, 1]
        else:
            dx = refer_path[i + 1, 0] - refer_path[i, 0]
            dy = refer_path[i + 1, 1] - refer_path[i, 1]
            ddx = refer_path[i + 1, 0] + refer_path[i - 1, 0] - 2 * refer_path[i, 0]
            ddy = refer_path[i + 1, 1] + refer_path[i - 1, 1] - 2 * refer_path[i, 1]
        refer_path[i, 2] = math.atan2(dy, dx)  # yaw
        # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
        refer_path[i, 3] = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))  # 曲率k计算
    return refer_path


def calc_track_error(refer_path, x, y):
    """计算跟踪误差
    Args:
        x (_type_): 当前车辆的位置x
        y (_type_): 当前车辆的位置y

    Returns:
        _type_: _description_
    """
    # 寻找参考轨迹最近目标点
    d_x = [refer_path[i, 0] - x for i in range(len(refer_path))]
    d_y = [refer_path[i, 1] - y for i in range(len(refer_path))]
    d = [np.sqrt(d_x[i] ** 2 + d_y[i] ** 2) for i in range(len(d_x))]
    s = np.argmin(d)  # 最近目标点索引

    yaw = refer_path[s, 2]
    k = refer_path[s, 3]
    angle = normalize_angle(yaw - math.atan2(d_y[s], d_x[s]))
    e = d[s]  # 误差
    if angle < 0:
        e *= -1

    return e, k, yaw, s


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    copied from https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/stanley_control/stanley_control.html
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def cal_Ricatti(A, B, Q, R):
    """解代数里卡提方程

    Args:
        A (_type_): 状态矩阵A
        B (_type_): 状态矩阵B
        Q (_type_): Q为半正定的状态加权矩阵, 通常取为对角阵；Q矩阵元素变大意味着希望跟踪偏差能够快速趋近于零；
        R (_type_): R为正定的控制加权矩阵，R矩阵元素变大意味着希望控制输入能够尽可能小。

    Returns:
        _type_: _description_
    """
    # 设置迭代初始值
    Qf = Q
    P = Qf
    # 循环迭代
    for t in range(N):
        P_ = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        if (abs(P_ - P).max() < EPS):
            break
        P = P_
    return P_


def state_space(vehicle, ref_delta, ref_yaw):
    v = total_speed(vehicle)
    A = np.matrix([
        [1.0, 0.0, -v * dt * math.sin(ref_yaw)],
        [0.0, 1.0, v * dt * math.cos(ref_yaw)],
        [0.0, 0.0, 1.0]])

    B = np.matrix([
        [dt * math.cos(ref_yaw), 0],
        [dt * math.sin(ref_yaw), 0],
        [dt * math.tan(ref_delta) / L,
         v * dt / (L * math.cos(ref_delta) * math.cos(ref_delta))]
    ])

    return A, B


def lqr(robot_state, refer_path, s0, A, B, Q, R):
    """
    LQR控制器
    """
    # x为位置和航向误差
    x = robot_state[0:3] - refer_path[s0, 0:3]

    P = cal_Ricatti(A, B, Q, R)

    K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    u = K @ x
    u_star = u  # u_star = [[v-ref_v,delta-ref_delta]]
    # print(u_star)
    return u_star[0, 1]


def lqr_controller(vehicle, path):
    path = np.array(path)
    path = MyReferencePath(path)
    x = vehicle.get_location().x
    y = vehicle.get_location().y
    psi = vehicle.get_transform().rotation.yaw
    psi = math.radians(psi)
    v = total_speed(vehicle)
    robot_state = np.zeros(4)
    robot_state[0] = x
    robot_state[1] = y
    robot_state[2] = psi
    robot_state[3] = v
    e, k, ref_yaw, s0 = calc_track_error(path, robot_state[0], robot_state[1])
    ref_delta = math.atan2(L * k, 1)
    A, B = state_space(vehicle, ref_delta, ref_yaw)
    delta = lqr(robot_state, path, s0, A, B, Q, R)
    delta = delta + ref_delta
    steer = math.tan(delta)
    steer = math.atan(steer)
    steer = math.degrees(steer) / 40
    if steer > 1:
        steer = 1
    elif steer < -1:
        steer = -1
    return steer


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
        if dis < 12 and total_v > total_v1:
            break
        time.sleep(t_s)

    # 变道
    total_v = total_speed(vehicle)
    target_speed_2 = 50 / 3.6
    v_error = target_speed_2 - total_v
    start_t = time.time()
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        location_vehicle1 = vehicle1.get_location()
        location_vehicle = vehicle.get_location()
        v_trans = vehicle.get_transform()
        waypoint = map.get_waypoint(location_vehicle1, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        waypoints = []
        waypoints.append(waypoint.next(0.5)[0].get_left_lane().transform.location)
        waypoints.append(waypoint.next(1.0)[0].get_left_lane().transform.location)
        waypoints.append(waypoint.next(1.5)[0].get_left_lane().transform.location)
        path = get_path(vehicle, waypoints)
        steer = lqr_controller(vehicle, path)
        throttle, brake, v_error = pid_control(vehicle, target_speed_2, v_error)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        if dis < 4:
            break
        time.sleep(t_s)

    # 持续超越
    total_v = total_speed(vehicle)
    target_speed_3 = 50 / 3.6
    v_error = target_speed_3 - total_v
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        location_vehicle1 = vehicle1.get_location()
        location_vehicle = vehicle.get_location()
        v_trans = vehicle.get_transform()
        waypoint = map.get_waypoint(location_vehicle, project_to_road=True, lane_type=carla.LaneType.Driving)
        waypoints = []
        waypoints.append(waypoint.next(8.0 + dis)[0].transform.location)
        waypoints.append(waypoint.next(9.0 + dis)[0].transform.location)
        waypoints.append(waypoint.next(10.0 + dis)[0].transform.location)
        path = get_path(vehicle, waypoints)
        steer = lqr_controller(vehicle, path)
        throttle, brake, v_error = pid_control(vehicle, target_speed_3, v_error)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        if dis > 10:
            break
        time.sleep(t_s)

    # 变回原道
    total_v = total_speed(vehicle)
    target_speed_4 = 50 / 3.6
    v_error = target_speed_4 - total_v
    while 1:
        dis_x, dis_y, dis = distance(vehicle, vehicle1)
        tracking(vehicle1, target_speed_v1)
        location_vehicle1 = vehicle1.get_location()
        location_vehicle = vehicle.get_location()
        v_trans = vehicle.get_transform()
        waypoint = map.get_waypoint(location_vehicle1, project_to_road=True, lane_type=carla.LaneType.Driving)
        waypoints = []
        waypoints.append(waypoint.next(8.0 + dis)[0].transform.location)
        waypoints.append(waypoint.next(9.0 + dis)[0].transform.location)
        waypoints.append(waypoint.next(10.0 + dis)[0].transform.location)
        path = get_path(vehicle, waypoints)
        steer = lqr_controller(vehicle, path)
        throttle, brake, v_error = pid_control(vehicle, target_speed_4, v_error)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
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
