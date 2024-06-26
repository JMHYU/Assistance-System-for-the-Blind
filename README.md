# Assistance System for the Blind using Object Detection (YOLOv7-tiny & Jetson Nano)
I designed and implemented a real-time pedestrian assistance system for visually impaired individuals,  utilizing Jetson Nano board. Fine-tuned YOLOv7 with a custom dataset, optimized it with TensorRT for real-time, on-device performance, and developed an auditory guidance system incorporating an object tracking algorithm. 
<br/> <br/> <br/>

## Motivation and Objectives
### 1. Motivation
Lately, there has been an increasing presence of various obstacles—such as shared scooters, bicycles, pedestrian protection bollards, delivery motorcycles, and various types of signage—on tactile paving blocks meant for the visually impaired in Korea. This unauthorized placement severely threatens the mobility rights of visually impaired individuals, particularly affecting those who are completely blind. These individuals face significant risks during navigation as they cannot identify these obstacles, even if they detect them, leading to considerable discomfort.

Existing pedestrian assistance systems for the visually impaired, found in the Jetson community projects, often fail to provide any contextual information. They merely alert the user about nearby objects and suggest a direction to avoid them, thus relegating visually impaired users to passive reliance on these systems. Previous projects employing depth prediction or semantic segmentation only measure the distance between pedestrians and obstacles but fall short in explaining how these objects interfere with the path, leaving a critical gap in context. Therefore, even with guidance, visually impaired individuals remain unable to gauge the size of the objects or the severity of the situation.

### 2. Objectives
The system needs to detect obstacles encountered on sidewalks covered with tactile blocks, hence it must be tailored to the local environment, including elements like signboards, air banners, flower pots, bollards, utility poles, shared scooters, and delivery motorcycles. Merely detecting all obstacles is not sufficient; the algorithm should be able to discern which situations pose a threat and alert the user through selective voice signals.

It is crucial for the algorithm to provide sufficient context so that visually impaired users can independently assess situations, recognize the level of danger, and plan their movements accordingly. The system should be capable of identifying the type of object and its state (whether it is within the region of interest or approaching).

To be practical for real-world usage, the entire system, including the CNN model and algorithm, must operate on an edge device like the Jetson Nano, functioning independently without internet connectivity. It should also be applicable to live walking videos captured by a camera, with processing speeds fast enough to maintain a frame rate of 15 FPS.

<br/><img width="80%" src="https://github.com/JMHYU/Assistance-System-for-the-Blind-using-Object-Detection/assets/165994759/11ae830f-d86b-45d6-a0de-4c590a7ea47b"/>
<br/> <br/> <br/>

## Technical contributions
### 1. Baseline
a) Transfer Learning YOLOv7-tiny model to make a custom model
- DataSet: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189 <br/>
(This dataset is open to Korean nationals only) <br/>
> Object classes(subcategory: 'movable objects'): Bicycle, Bus, Car, Carrier, Cat, Dog, Motorcycle, Movable Signage, Person, Scooter, Stroller, Truck, Wheelchair <br/>
> Object classes(subcategory: 'fixed object objects'): Barricade, Bench, Bollard, Chair, Fire Hydrant, Kiosk, Parking Meter, Pole, Potted Plant, Power Controller, Stop, Table, Traffic Light, Traffic Light Controller, Traffic Sign, Tree Trunk <br/>

- Training: used the official YOLOv7 Github Repository (https://github.com/WongKinYiu/yolov7) <br/>
(Important: Instead of using cfg/training/yolov7-tiny.yaml, use cfg/deploy/yolov7-tiny.yaml while training)
I had to edit the number of class 'nc' from 80 to 29 (the dataset I am using has 29 classes) in yolov7-tiny.yaml


b) Building TensorRT Engine on Jetson Nano
I converted a YOLOv7-tiny custom model into a TRT engine using the procedure outlined on Github at JetsonYoloV7-TensorRT (https://github.com/mailrocketsystems/JetsonYoloV7-TensorRT).
