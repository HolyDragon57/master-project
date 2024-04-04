### 标注公司沟通

* 统一标注类别文件，对齐不同类别区分标准
* 要求有double check，保证数据可用性（我们只做最终抽样检查）
* 标注好的文件统一命名规范（用时间区分）上传到百度网盘固定文件夹，原始视频和标注json文件需要一一对应
* 每个动作长度最好不能少于2秒
* 动作初始帧标注框需要有标签表示动作，同时需要划分时间轴
* 尽量不要在人物在画面边缘时画框

## 数据流转

1. 标注公司上传百度网盘，文件目录结构如下

   ```text
   |-- 标注数据
   	|-- 20240307-自然语言处理
   		|-- xxx.mp4
   		|-- xxx.json
   		|-- 02.mp4
   		|-- 02.json
   		...
   	|-- 20240308-课程名
   	...
   ```

   注意事项：

   * 一级文件夹由`八位标注时间`-`课程名`构成，标注时间是为了方便排序查看哪些是后续添加数据
   * 二级文件夹中需要**保证mp4视频文件名和标注json文件名有一一对应的部分**，最好文件名相同

2. 从百度网盘上按照对应结构下载数据到`master-project/data-preprocess/annotated_data`文件夹下

3. 运行`preprocess.py`增量处理添加数据，预处理数据存放在`master-project/data-preprocess/extraction_result`下

   ```text
   |--	extraction_result
   	|-- 20240307-自然语言处理
   		|-- 01-看书.mp4
   		|-- 02-巡查.mp4
   		|-- ...
           |-- student_clip-action.csv
           |-- teacher_clip-action.csv
   	|-- 20240308-课程名
   		|-- ...
   	|-- student_clip-action.csv
   	|-- teacher_clip-action.csv
   ```

4. 检查标注数据质量

   * 检查segment和人体标注框数量是否相等：进而判定标注框能否被ByteTrack识别（IoU>0.2）
   * 人工点开视频（随机抽样）检查标注框跟踪情况；检查标注框内动作是否为对应动作
   * 至少每种类别500条数据

5. 数据备份

   * 将新处理好的文件夹上传到百度网盘的`clip-action数据集`中；更新记录所有标签的`student_clip-action.csv`文件和`teacher_clip-action.csv`文件

### data-preprocess

为提升标注框质量，需要用ByteTrack标注框检测IoU替代

裁切方式：空间上动作发生时间内焦点人物最大的长宽+bounding box绘制area of interest；时间上则与标注文件对应

构建clip-action dataset

* 会有邻近标注框混淆情况，即人物替换
  * 21，25跟丢
* 人物在画面边缘ByteTrack未识别导致数据不可用
  * 有一段老师巡查不可用

问题

* 13看书
* 14与同学交流
* 16看书
* 23记笔记
* 最后一个选取的学生对象占画面太小了
