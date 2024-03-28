## Getting Started

We provide a script `train_net_syncvis.py`, that is made to train all the configs in syncvisFormer.

To train a model with "train_net_syncvis.py" on VIS, first setup the corresponding datasets following the instructions in folder `datasets`

Then run with COCO pretrained weights on r-50 config:
```
python train_net_syncvis.py --num-gpus 8 --config-file configs/youtubevis_2019/syncvis_R50_bs8.yaml MODEL.WEIGHTS r50_coco.pth
```

To evaluate model's performace, run with:
```
python train_net_syncvis.py --num-gpus 8 --config-file configs/youtubevis_2019/syncvis_R50_bs8.yaml --eval-only MODEL.WEIGHTS path/to/ckpt
```


