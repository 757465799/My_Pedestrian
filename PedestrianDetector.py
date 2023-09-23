import atexit
import threading
import time
import carla
import random
import cv2
import numpy as np
import torch
import albumentations as A
import torch.nn.functional as F
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from Model.DETR_backbone import BackboneAndPos
from Model.DETR_transformer import Transformer
from Model.DETR_Main import DETR, box_x1y1x2y2

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Carla:

    def __init__(self, autodrive, random_spawn_point):
        self.stop_simulator = False
        self.extrinsic = None
        self.intrinsic = None
        self.image_size_y = 720
        self.image_size_x = 1280
        self.transform_c = None
        self.world = world
        self.walkers = []
        self.controllers = []
        self.route_index = None
        self.route = None
        self.frame = None
        self.autopilot = autodrive
        self.spawn_point = random_spawn_point
        self.vehicle = None
        self.sensor = None
        self.running = True

        backbone = BackboneAndPos()
        transformer = Transformer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DETR_Model = DETR(backbone, transformer)
        self.DETR_Model = self.DETR_Model.to(self.device)

        # 保存原始权重
        original_weights = {name: param.clone() for name, param in self.DETR_Model.named_parameters()}

        # 加载权重
        weights_path = 'DETR_best_weights.pth'

        self.DETR_Model.load_state_dict(torch.load(weights_path))
        # for layer_name in torch.load(weights_path).keys():
        #     print(layer_name)
        # 比较权重
        for name, param in self.DETR_Model.named_parameters():
            assert not torch.equal(original_weights[name], param), f"权重 {name} 没有改变！"

        print("权重加载成功！")

        self.DETR_Model.eval()

        self.base_environment()
        self.PedestrianDetector()

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
        # 随机选择一个天气类型
        weather = random.choice(weather_presets)
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
                print("当前生成的行人数量：", len(self.walkers))
            except RuntimeError:
                continue

        def random_walk(walkers):
            while self.running:
                while True:
                    # Control the movement of pedestrians
                    for walker in walkers:
                        walker_control = carla.WalkerControl()
                        walker_control.speed = 1.0 + random.random()
                        walker_control.direction = carla.Vector3D(x=random.choice([1.0, -1.0]),
                                                                  y=random.choice([1.0, -1.0]))
                        walker.apply_control(walker_control)
                    time.sleep(3)  # Change direction every 3 second

        # Start the random walk in a new thread
        threading.Thread(target=random_walk, args=(self.walkers,)).start()

    def PedestrianDetector(self):
        confidence_threshold = 0.5
        while self.running:
            if self.frame is not None:
                frame_copy = self.frame.copy()

                # 使用模型进行推理
                outputs = detect_pedestrians(self.DETR_Model, self.frame)

                # 假设 outputs 包括 'pred_logits' 和 'pred_boxes'
                pred_logits = outputs['pred_logits']
                pred_boxes = outputs['pred_boxes']

                # 计算类别概率
                probabilities = F.softmax(outputs['pred_logits'], dim=-1)

                # 获取行人类别的概率（假设索引0表示行人）
                pedestrian_probabilities = probabilities[:, :, 1]

                # 应用置信度阈值
                mask = pedestrian_probabilities > confidence_threshold

                # 获取选定的边界框
                selected_boxes = outputs['pred_boxes'][mask]

                # 检查是否找到了行人边界框
                if selected_boxes.shape[0] > 0:
                    print("Pedestrian bounding box found！")
                else:
                    print("Pedestrian bounding box not found")

                height, width, _ = self.frame.shape

                # 将中心宽高格式的边界框转换为左上和右下坐标格式
                selected_boxes = box_x1y1x2y2(selected_boxes)

                # 将坐标转换为实际像素值（确保转换因子在相同的设备上）
                conversion_factor = torch.Tensor([width, height, width, height]).to(selected_boxes.device)
                selected_boxes = (selected_boxes * conversion_factor).cpu().numpy()

                # 清除当前图像
                plt.clf()

                for box in selected_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 使用OpenCV显示图像
                cv2.imshow('my car', frame_copy)
                key = cv2.waitKey(1)  # 更新窗口并等待1ms

                # 如果用户按下'q'键，退出循环
                if key == ord('q'):
                    self.running = False

        # 退出循环后，关闭所有OpenCV窗口
        cv2.destroyAllWindows()

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


def detect_pedestrians(model, image):
    processed_image = preprocess_image(image)  # 假设 images 包含一个图像
    processed_image = processed_image.to(device)  # 转移到设备
    processed_image = [processed_image]  # 将张量封装在列表中

    # 通过模型执行推理
    with torch.no_grad():
        outputs = model(processed_image)  # 输入是一个张量列表

    return outputs


def preprocess_image(images):
    # Convert the image to NumPy array
    np_image = np.array(images)  # Assuming images is a list with one element

    # Define the transformation pipeline
    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

    processed_image = transform(image=np_image)['image']
    processed_image = processed_image  # 添加批次维度
    return processed_image


if __name__ == '__main__':
    Carla(True, False)

cv2.destroyAllWindows()  # 添加这行代码