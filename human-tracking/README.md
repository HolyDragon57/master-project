### 执行命令

> 两种不同大小的模型

```bash
python src/human_tracking.py video -f src/yolox_m_mix_det.py -c pretrained/bytetrack_m_mot17.pth.tar --input_path ../demo.mp4 --output_path ../
python src/human_tracking.py video -f src/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --input_path ../demo.mp4 --output_path ../
```

### 参数说明

输出格式：`{frame}, {id}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {confidence}, {x}, {y}, {z}`

> 坐标(x, y, z)在2D问题中可被忽略（用-1, -1, -1替代）。同理，bounding box信息在3D问题中被忽略

`--fp16`使用混合精度

`--fuse`融合卷积层和归一化层，在推理阶段提高模型性能和效率

`--save_result`将追踪结果（即上述输出）记录下来

`--trt`使用TensorRT加速

> 以上均已默认设置

`--input_path`指明输入视频路径信息

`--output_path`指明输出标签文件路径

### 环境配置

```bash
pip install cython cython_bbox pycocotools
pip install -r requirements.txt
```

### 注意事项

* requirements.txt里onnx和onnxruntime有版本问题
* numpy有版本问题
  * yolox -> tracker -> byte_tracker.py and matching.py: np.float => float
  * 或者使用1.22.4以下版本
* Python不同版本下模型运行速度不同

### 思考

一种相对来说更不失真的方式：blur/black the background；

甚至可以根本不用处理原视频，只需要加上bounding box绘图

存在一定程度的覆盖遮挡问题

连续性按道理比较好，场景并不复杂

睡觉时没检测出来是个人，出镜头入镜头会重新设置ID

有时候不得不容忍缺陷然后继续前进
