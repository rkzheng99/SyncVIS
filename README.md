# SyncVIS: Synchronized Video Instance Segmentation

## :sunny:Overview

**SyncVIS** explicitly introduces video-level query embeddings and designs two key modules to synchronize video-level query with frame-level query embeddings: a **synchronized video-frame modeling paradigm** and a **synchronized embedding optimization strategy**. The former attempts to promote the mutual learning of frame- and video-level embeddings with each other and the latter divides large video sequences into small clips for easier optimization. In this page, we provide further experiments of our approaches and additional visualizations including both specific scenarios and failure cases as well as their analysis.

![image](https://github.com/rkzheng99/SyncVIS/blob/main/pics/model.png)

## :pencil2:Further Experiments

We list the results of building our method upon other popular VIS methods apart from IDOL and VITA. Worth mentioning, TMT-VIS is mainly designed for training on multiple datasets, and in our experiments, we mainly test the effectiveness of our model when training on a single YTVIS-19 dataset. 

**Table 1 Experiments on aggregating our design to current VIS methods (ResNet-50)**
|Method|AP|Method|AP|
|:----|:----|:----|:----|
|Mask2Former|45.1|VITA|49.5|
|+ Synchronized Video-Frame Modeling|50.3|+ Synchronized Video-Frame Modeling|53|
|+ Synchronized Embedding Optimization|46.7|+ Synchronized Embedding Optimization|51.2|
|+ Both (SyncVIS)|51.5|+ Both (SyncVIS)|54.2|
|TMT-VIS|47.3|DVIS|52.6|
|+ Synchronized Video-Frame Modeling|51.1|+ Synchronized Video-Frame Modeling|54.9|
|+ Synchronized Embedding Optimization|48.7|+ Synchronized Embedding Optimization|54|
|+ Both (SyncVIS)|51.9|+ Both (SyncVIS)|55.8|
|GenVIS|51.3|IDOL|49.5|
|+ Synchronized Video-Frame Modeling|54.4|+ Synchronized Video-Frame Modeling|55.1|
|+ Synchronized Embedding Optimization|52.7|+ Synchronized Embedding Optimization|51.3|
|+ Both (SyncVIS)|55.4|+ Both (SyncVIS)|56.5|

## :sparkles:Visualization

### Fast-Moving Instances
In this part, we present you several cases showing that our model is capable of tracking and segmenting instances with greater velocity. These results demonstrate that with our video-frame synchronization, SyncVIS is able to depict the trajectories and appearances of these fast-moving objects. 

#### Racing Car
We demonstrate that our SyncVIS shows the ability of segmenting and tracking fast-moving racing cars with precision and consistency.

<img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_0.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_1.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_2.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_3.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_4.jpg" width="300px">

#### Skateboarding

We demonstrate that our SyncVIS shows the ability of segmenting and tracking fast-moving man skating on his skateboard, segmenting the man's pose and movement with precision and consistency.

<img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/skating_0.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/skating_1.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/skating_2.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/skating_3.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/skating_4.jpg" width="300px"> 
 



### Failure Cases
![image](https://github.com/rkzheng99/SyncVIS/blob/main/pics/failure.png)

As for limitations, our model has problem in segmenting very crowded or heavily occluded scenarios. Even though our model shows better performance in segmenting complex scenes with multiple instances and occlusions than previous approaches. 


