import atexit
import os
import threading
import time
import carla
import random
import cv2
import numpy as np

# 首先创建连接
client = carla.Client('localhost', 2000)
# 也可以使用client.load_word('map')可加载不同的地图
world = client.get_world()

try:
    world = client.load_world('Town06')
except RuntimeError:
    print('loading world Town')
    # 设置time_out
    client.set_timeout(50.0)

# 清理地图中的所有车辆、行人和传感器
for actor in world.get_actors():
    if 'vehicle' in actor.type_id or 'walker' in actor.type_id or 'sensor' in actor.type_id:
        actor.destroy()


class Carla:

    def __init__(self, autodrive, random_spawn_point):
        self.stop_simulator = False
        self.extrinsic = None
        self.intrinsic = None
        self.image_size_y = None
        self.image_size_x = None
        self.transform_c = None
        self.world = world
        self.walkers = []
        # self.add_vehicles(30)
        self.controllers = []
        self.route_index = None
        self.route = None
        self.frame = None
        self.autopilot = autodrive
        self.spawn_point = random_spawn_point
        self.vehicle = None
        self.sensor = None
        self.running = True
        self.base_environment()
        self.data_collect()

    # sensor的回调函数，主要进行一些简单的数据处理
    def data_process(self, image):
        # frombuffer将data以流的形式读入转化成ndarray对象
        image = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        image = image.reshape(720, 1280, 4)
        self.frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    def base_environment(self):
        # 创建一个包含所有预设天气类型的列表
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.WetCloudyNoon,
            carla.WeatherParameters.MidRainyNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.SoftRainNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.WetSunset,
            carla.WeatherParameters.WetCloudySunset,
            carla.WeatherParameters.MidRainSunset,
            carla.WeatherParameters.HardRainSunset,
            carla.WeatherParameters.SoftRainSunset,
        ]
        weather = weather_presets[12]
        print(weather)
        # 随机选择一个天气类型
        # weather = random.choice(weather_presets)
        # 设置天气
        self.world.set_weather(weather)

        # 车辆蓝图和传感器蓝图
        blueprint_library = world.get_blueprint_library()
        vehicle = blueprint_library.filter('model3')[0]
        camera = blueprint_library.find('sensor.camera.rgb')
        self.image_size_x = 1280
        self.image_size_y = 720

        # 设置camera的属性
        camera.set_attribute('image_size_x', f'{self.image_size_x}')
        camera.set_attribute('image_size_y', f'{self.image_size_y}')
        camera.set_attribute('fov', '70')
        self.transform_c = carla.Transform(carla.Location(x=0.4, z=1.2))

        # 循环，直到车辆成功生成
        while self.vehicle is None:
            # 选择一个随机的生成点
            transform_v = random.choice(world.get_map().get_spawn_points())
            # 尝试在该生成点生成车辆
            self.vehicle = world.try_spawn_actor(vehicle, transform_v)

        # 设置model3的驾驶模式
        self.vehicle.set_autopilot(self.autopilot)

        # 生成并绑定相机传感器
        self.sensor = world.spawn_actor(camera, self.transform_c, attach_to=self.vehicle)
        self.sensor.listen(self.data_process)

        # 相机内参和外参
        # self.intrinsic, self.extrinsic = self.get_camera_intrinsic_and_extrinsic(camera)

        # 设置观察者的位置
        spectator = world.get_spectator()
        self.transform_c = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(self.transform_c.location + carla.Location(x=-10, z=10),
                                                carla.Rotation(pitch=-40)))

        walker_blueprints = blueprint_library.filter('walker.*')

        # Ensure there are at least 10 different walker blueprints
        assert len(walker_blueprints) >= 40, "Not enough walker blueprints to create unique walkers."

        for i in range(40):
            # Select a unique walker blueprint for each pedestrian
            walker_bp = walker_blueprints[i]
            spawn_point = random.choice(world.get_map().get_spawn_points())
            try:
                walker = world.spawn_actor(walker_bp, spawn_point)
                self.walkers.append(walker)
            except RuntimeError:
                continue

        def random_walk(walkers):
            print('number of pedestrian', len(walkers))
            while self.running:
                while True:
                    # Control the movement of pedestrians
                    for walker in walkers:
                        walker_control = carla.WalkerControl()
                        walker_control.speed = 1.0 + random.random()
                        walker_control.direction = carla.Vector3D(x=random.choice([1.0, -1.0]),
                                                                  y=random.choice([1.0, -1.0]),
                                                                  z=random.choice([1.0, -1.0]))
                        walker.apply_control(walker_control)
                    time.sleep(3)  # Change direction every 3 second

        # Start the random walk in a new thread
        threading.Thread(target=random_walk, args=(self.walkers,)).start()

    def data_collect(self):
        self.running = True
        vehicle = self.vehicle
        count = 0
        dataset_size = 1000
        # save_path = './Dataset/'
        save_path = './'
        while self.running:
            if self.frame is not None:
                # 当车辆遇到交通灯时，设置交通灯为绿
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    traffic_light.set_state(carla.TrafficLightState.Green)
                # 显示实时结果
                cv2.namedWindow('my car', cv2.WINDOW_NORMAL)
                cv2.imshow('my car', cv2.resize(self.frame, (800, 600)))
                cv2.waitKey(50)

                # frame = cv2.resize(self.frame, (512, 512))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.frame
                if count % 5 == 0:

                    # Save frame as an image
                    image_name = f"HardRainSunset_frame4_{count}.jpg"  # Unique image name based on the count
                    image_path = os.path.join(save_path, image_name)
                    cv2.imwrite(image_path, frame)
                count += 1

                if count >= dataset_size:
                    self.running = False
                    break

        # 在程序结束时销毁行人和控制器
        def destroy():
            self.sensor.destroy()
            self.vehicle.destroy()
            for controllers in self.controllers:
                controllers.stop()
                controllers.destroy()
            for walkers in self.walkers:
                walkers.destroy()

        atexit.register(destroy)

        if cv2.waitKey(1) == ord(' '):
            self.stop_simulator = True


if __name__ == '__main__':
    Carla(True, False)

cv2.destroyAllWindows()  # 添加这行代码
