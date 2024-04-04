### Label file format

`{frame_id}, {human_id}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {action_label}`

可以抽象出一层tracklet代表clip

汇总标注分为两部分：

1. 同一个clip不同空间分割得到的各个人物的动作汇总到一个clip
   1. <video_name>-clip_1_1：视频片段tracklet，人物追踪标签，动作识别logits
   2. video-clip_1_2
2. 不同时间的clip重叠变成一个视频

### 参数说明

* clip_duration必须能整除time_step
* 由于不同视频视角不同， 人物站姿坐姿不尽相同，因此upper_body_ratio可以根据具体情况设置
* 
