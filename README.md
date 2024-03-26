# SyncVIS: Synchronized Video Instance Segmentation

Recent DETR-based methods have advanced the development of Video Instance Segmentation (VIS) through transformers' efficiency and capability in modeling spatial and temporal information. Despite harvesting remarkable progress, existing works follow asynchronous designs, which model video sequences via either video-level queries only or adopting query-sensitive cascade structures, resulting in difficulties when handling complex and challenging video scenarios. In this work, we analyze the cause of this phenomenon and the limitations of the current solutions, and propose to conduct synchronized modeling via a new framework named **SyncVIS**. Specifically, SyncVIS explicitly introduces video-level query embeddings and designs two key modules to synchronize video-level query with frame-level query embeddings: a synchronized video-frame modeling paradigm and a synchronized embedding optimization strategy. The former attempts to promote the mutual learning of frame- and video-level embeddings with each other and the latter divides large video sequences into small clips for easier optimization.

## Further Experiments

We list the results of building our method upon other popular VIS methods apart from IDOL and VITA. Worth mentioning, TMT-VIS is mainly designed for training on multiple datasets, and in our experiments we mainly test the effectiveness of our model when training on a single YTVIS-19 dataset.

Table 1 Experiments on aggregating our design to current VIS methods
| Method                                | Backbone  | AP |
|---------------------------------------|-----------|----|
| GenVIS                                | ResNet-50 |51.3|
| + Synchronized Video-Frame Modeling   | ResNet-50 |53.1|
| + Synchronized embedding optimization | ResNet-50 |52.6|
| + All                                 | ResNet-50 |54.2|
| TMT-VIS                               | ResNet-50 |47.3|
| + Synchronized Video-Frame Modeling   | ResNet-50 |50.6|
| + Synchronized embedding optimization | ResNet-50 |48.7|
| + All                                 | ResNet-50 |51.5|
| DVIS                                  | ResNet-50 |52.6|
| + Synchronized Video-Frame Modeling   | ResNet-50 |53.9|
| + Synchronized embedding optimization | ResNet-50 |53.4|
| + All                                 | ResNet-50 |54.1|

## Visualization

### Fast-Moving Instances
In this part, we present you several cases showing that our model is capable of tracking and segmenting instances with greater velocity.

#### Racing Car
We demonstrate that our SyncVIS shows the ability of segmenting and tracking fast-moving racing cars with precision and consistency.

<img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_0.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_1.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_2.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_3.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_4.jpg" width="300px">

#### Skateboarding

### Failure Cases

As for limitations, our model has problem in segmenting very crowded or heavily occluded scenarios. Even though our model shows better performance in segmenting complex scenes with multiple instances and occlusions than previous approaches. 


