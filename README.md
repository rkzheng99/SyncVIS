# SyncVIS: Synchronized Video Instance Segmentation

Recent DETR-based methods have advanced the development of Video Instance Segmentation (VIS) through transformers' efficiency and capability in modeling spatial and temporal information. Despite harvesting remarkable progress, existing works follow asynchronous designs, which model video sequences via either video-level queries only or adopting query-sensitive cascade structures, resulting in difficulties when handling complex and challenging video scenarios. In this work, we analyze the cause of this phenomenon and the limitations of the current solutions, and propose to conduct synchronized modeling via a new framework named **SyncVIS**. Specifically, SyncVIS explicitly introduces video-level query embeddings and designs two key modules to synchronize video-level query with frame-level query embeddings: a synchronized video-frame modeling paradigm and a synchronized embedding optimization strategy. The former attempts to promote the mutual learning of frame- and video-level embeddings with each other and the latter divides large video sequences into small clips for easier optimization.

## Visualization

### Fast-Moving Instances
In this part, we present you several cases showing that our model is capable of tracking and segmenting instances with greater velocity.

#### Racing Car
We demonstrate that our SyncVIS shows the ability of segmenting and tracking fast-moving racing cars with precision and consistency.

<img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_0.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_1.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_2.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_3.jpg" width="300px"> <img src="https://github.com/rkzheng99/SyncVIS/blob/main/pics/fast_moving/racing_car_4.jpg" width="300px">

#### Skateboarding

### Failure Cases

As for limitations, our model has problem in segmenting very crowded or heavily occluded scenarios. Even though our model shows better performance in segmenting complex scenes with multiple instances and occlusions than previous approaches. 


