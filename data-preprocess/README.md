### 标注公司沟通

* 统一标注类别文件，对齐不同类别区分标准
* 要求有double check，保证数据可用性（我们只做最终抽样检查）
* 标注好的文件统一命名规范（用时间区分）上传到百度网盘固定文件夹，原始视频和标注json文件需要一一对应

### data-preprocess

为提升标注框质量，需要用ByteTrack标注框检测IoU替代

裁切方式：空间上动作发生时间内焦点人物最大的长宽+bounding box绘制area of interest；时间上则与标注文件对应

构建clip-action dataset

