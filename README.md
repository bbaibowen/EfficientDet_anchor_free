# EfficientDet_anchor_free
EfficientDet_anchor_free



# Introduction
This is EfficientDet anchor-free in Pytorch,i also completed EfficientDet_anchor-based.


## Results
| |This report(anchor-free)| This report(anchor-based)|Paper   |
| :-----  | :-----    | :------ |:------ |
|network|Efficientnet-b0|Efficientnet-b0|Efficientnet-b0|
|datasets|COCO2017|VOC0712|COCO2017|
|notes|Multi-scales|mixup-up,label smooth,giou loss,cosine lr|Multi-scales|
|MAPS|32.9|68.5|32.4|
|b1-b7|TODO|


There are some problems in EfficientDet_anchor-based，which has low Maps and slow speed. I will fix it and share the code.


## Reference
  [FOCS](https://github.com/tianzhi0549/FCOS)
  [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf)



