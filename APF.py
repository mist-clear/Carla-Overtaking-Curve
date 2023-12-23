import copy
import math

import carla
import keyboard
import numpy as np
import pygame

actor_list = []


# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


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


def apf(ego, vehicle_list):
    ego_loc = ego.get_location()
    ego_vel = ego.get_velocity()
    # 获取车辆当前所在的道路
    waypoint = map.get_waypoint(ego_loc)
    # 获取道路中心点的位置
    road = waypoint.transform.location.y
    position = np.zeros((len(vehicle_list) * 8, 4))
    for i in range(len(vehicle_list)):
        # 设置障碍物形状参数（假设障碍物为矩形）
        obstacle_length = 5  # 障碍物长度
        obstacle_width = 1.875  # 障碍物宽度

        # 计算矩形区域的边界点坐标
        obstacle_points = [
            carla.Location(x=vehicle_list[i].get_location().x + obstacle_length / 2 + 3,
                           y=vehicle_list[i].get_location().y,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x,
                           y=vehicle_list[i].get_location().y + obstacle_width / 2,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x - obstacle_length / 2,
                           y=vehicle_list[i].get_location().y,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x,
                           y=vehicle_list[i].get_location().y - obstacle_width / 2,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x - obstacle_length / 2,
                           y=vehicle_list[i].get_location().y - obstacle_width / 2,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x + obstacle_length / 2 + 3,
                           y=vehicle_list[i].get_location().y - obstacle_width / 2,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x + obstacle_length / 2 + 3,
                           y=vehicle_list[i].get_location().y + obstacle_width / 2,
                           z=vehicle_list[i].get_location().z),
            carla.Location(x=vehicle_list[i].get_location().x - obstacle_length / 2,
                           y=vehicle_list[i].get_location().y + obstacle_width / 2,
                           z=vehicle_list[i].get_location().z)
        ]

        # 将矩形区域的8个边界点坐标添加到position数组
        for j in range(8):
            position[i * 8 + j][0] = obstacle_points[j % 8].x
            position[i * 8 + j][1] = obstacle_points[j % 8].y
            position[i * 8 + j][2] = vehicle_list[i].get_velocity().x
            position[i * 8 + j][3] = vehicle_list[i].get_velocity().y

    # 初始化车的参数
    d = 3.5  # 道路标准宽度

    W = 2.875  # 汽车宽度

    L = 4.7  # 车长

    P0 = np.array([ego_loc.x, ego_loc.y, ego_vel.x, ego_vel.y])  # 车辆起点位置，分别代表x,y,vx,vy

    waypoint = map.get_waypoint(vehicle_list[0].get_location(), project_to_road=True,
                                lane_type=carla.LaneType.Driving).next(15.0)[0].transform.location
    Pg = np.array([waypoint.x, waypoint.y, 0, 0])  # 目标位置

    # 障碍物位置
    Pobs = position

    P = np.vstack((Pg, Pobs))  # 将目标位置和障碍物位置合放在一起

    Eta_att = 5  # 引力的增益系数

    Eta_rep_ob = 10  # 斥力的增益系数

    Eta_rep_edge = 0.0  # 道路边界斥力的增益系数

    d0 = 30  # 障碍影响的最大距离

    num = P.shape[0]  # 障碍与目标总计个数

    len_step = 0.15  # 步长

    n = 1

    Num_iter = 100  # 最大循环迭代次数

    path = []  # 保存车走过的每个点的坐标
    delta = np.zeros((num, 2))  # 保存车辆当前位置与障碍物的方向向量，方向指向车辆；以及保存车辆当前位置与目标点的方向向量，方向指向目标点
    dists = []  # 保存车辆当前位置与障碍物的距离以及车辆当前位置与目标点的距离
    unite_vec = np.zeros((num, 2))  # 保存车辆当前位置与障碍物的单位方向向量，方向指向车辆；以及保存车辆当前位置与目标点的单位方向向量，方向指向目标点

    F_rep_ob = np.zeros((len(Pobs), 2))  # 存储每一个障碍到车辆的斥力,带方向
    v = np.linalg.norm(P0[2:4])  # 设车辆速度为常值
    # ***************初始化结束，开始主体循环******************
    Pi = P0[0:2]  # 当前车辆位置
    # count=0

    for i in range(Num_iter):
        # Check if the distance between current position (Pi) and goal position (Pg) is less than 1
        if ((Pi[0] - Pg[0]) ** 2 + (Pi[1] - Pg[1]) ** 2) ** 0.5 < 1:
            break
        dists = []
        path.append(Pi)

        # Calculate unit direction vectors from the vehicle to obstacles
        for j in range(len(Pobs)):
            delta[j] = Pi[0:2] - Pobs[j, 0:2]
            dists.append(np.linalg.norm(delta[j]))
            unite_vec[j] = delta[j] / dists[j]

        # Calculate unit direction vector from the vehicle to the goal
        delta[len(Pobs)] = Pg[0:2] - Pi[0:2]
        dists.append(np.linalg.norm(delta[len(Pobs)]))
        unite_vec[len(Pobs)] = delta[len(Pobs)] / dists[len(Pobs)]

        # Calculate attractive force
        F_att = Eta_att * dists[len(Pobs)] * unite_vec[len(Pobs)]

        # Calculate repulsive force
        for j in range(len(Pobs)):
            if dists[j] >= d0:
                F_rep_ob[j] = np.array([0, 0])
            else:
                F_rep_ob1_abs = Eta_rep_ob * (1 / dists[j] - 1 / d0) * (dists[len(Pobs)]) ** n / dists[j] ** 2
                F_rep_ob1 = F_rep_ob1_abs * unite_vec[j]
                F_rep_ob2_abs = n / 2 * Eta_rep_ob * (1 / dists[j] - 1 / d0) ** 2 * (dists[len(Pobs)]) ** (n - 1)
                F_rep_ob2 = F_rep_ob2_abs * unite_vec[len(Pobs)]
                F_rep_ob[j] = F_rep_ob1 + F_rep_ob2

        F_rep_edge = 0

        # Add repulsive force from road boundaries based on the vehicle's current position
        if - d + W / 2 < Pi[1] - road <= - d / 2:
            F_rep_edge = [0, Eta_rep_edge * v * np.exp(-d / 2 - Pi[1])]
        elif - d / 2 < Pi[1] - road <= - W / 2:
            F_rep_edge = np.array([0, 1 / 3 * Eta_rep_edge * Pi[1] ** 2])
        elif W / 2 < Pi[1] - road <= d / 2:
            F_rep_edge = np.array([0, - 1 / 3 * Eta_rep_edge * Pi[1] ** 2])
        elif d / 2 < Pi[1] - road <= d - W / 2:
            F_rep_edge = np.array([0, Eta_rep_edge * v * (np.exp(Pi[1] - d / 2))])

        # Calculate total force and direction
        F_rep = np.sum(F_rep_ob, axis=0) + F_rep_edge
        F_sum = F_att + F_rep

        UnitVec_Fsum = 1 / np.linalg.norm(F_sum) * F_sum

        # Calculate the next position of the vehicle
        Pi = copy.deepcopy(Pi + len_step * UnitVec_Fsum)

    return path


# Recursive implementation of the Bezier curve
def bezier(Ps, n, t):
    if n == 1:
        return Ps[0]
    return (1 - t) * bezier(Ps[0:n - 1], n - 1, t) + t * bezier(Ps[1:n], n - 1, t)


# Function to generate a path using Bezier curve based on reference points
def get_path(ref_path):
    points = np.zeros((4, 2))
    for i in range(4):
        points[i] = ref_path[int(i * len(ref_path) / 4)]

    ref = np.array([
        [points[0][0], points[0][1]],
        [points[1][0], points[1][1]],
        [points[2][0], points[2][1]],
        [points[3][0], points[3][1]]
    ])

    path = []  # Store path points
    # Generate Bezier curve
    for t in np.arange(0, 1.0, 0.01):
        p_t = bezier(ref, len(ref), t)
        path.append(p_t)

    path = np.array(path)
    path.tolist()
    paths = []

    # Convert path points to carla.Location format
    for i in range(len(path)):
        paths.append(carla.Location(x=path[i][0], y=path[i][1]))

    # Draw lines between consecutive path points for visualization
    for i in range(len(paths) - 1):
        world.debug.draw_line(paths[i], paths[i + 1], thickness=0.1,
                              life_time=0.1, color=carla.Color(b=255))

    return paths


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

    x, y, z = float(-420.0), float(12.8), float(1.0)
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
    distance_behind = 15.0  # 距离主车辆后方的距离

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
    ego = world.spawn_actor(ev_bp, spawn_point_vehicle)
    actor_list.append(ego)
    ego.set_light_state(carla.VehicleLightState.HighBeam)

    vehicles = [ego, vehicle1]
    target_speeds = [50 / 3.6, 40 / 3.6]
    # 生成监视器
    spectator = world.get_spectator()
    transform = vehicle1.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))

    # Initialise the camera floating behind the vehicle
    camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)

    # Start camera with PyGame callback
    camera.listen(lambda image: pygame_callback(image, renderObject))

    # Get camera dimensions
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()

    # Instantiate objects for rendering and vehicle control
    renderObject = RenderObject(image_w, image_h)

    # Initialise the display
    pygame.init()
    gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # Draw black to the display
    gameDisplay.fill((0, 0, 0))
    gameDisplay.blit(renderObject.surface, (0, 0))
    pygame.display.flip()

    vehicle1.set_target_velocity(carla.Vector3D(-40 / 3.6, 0, 0))
    ego.set_target_velocity(carla.Vector3D(-50 / 3.6, 0, 0))

    total_v = total_speed(ego)
    target_speed_ego = 50 / 3.6
    v_error = target_speed_ego - total_v
    while 1:
        # Advance the simulation time
        world.tick()
        # Update the display
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()
        for vehicle, target_speed in zip(vehicles[1:], target_speeds[1:]):
            tracking(vehicle, target_speed)
        v_trans = ego.get_transform()
        location_vehicle = ego.get_location()
        tar_loc = apf(ego, vehicles[1:])
        path = get_path(tar_loc)
        tar_loc = path[50]
        steer = pure_pursuit(tar_loc, v_trans)
        throttle, brake, v_error = pid_control(ego, target_speed_ego, v_error)
        ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        world.debug.draw_line(location_vehicle, location_vehicle, thickness=0.1, life_time=20, color=carla.Color(r=255))
        # Stop and remove the camera
        camera.stop()
        camera.destroy()

        # Spawn new camera and attach to new vehicle
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)
        camera.listen(lambda image: pygame_callback(image, renderObject))

        # Update PyGame window
        gameDisplay.fill((0, 0, 0))
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()
        if keyboard.is_pressed('x'):
            break

finally:
    if len(actor_list) == 0:
        print("Finish!")
    else:
        for actor in actor_list:
            actor.destroy()
        print("Finish!")
