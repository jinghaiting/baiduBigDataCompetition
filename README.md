### 项目介绍

详情请看此链接：[<http://www.ikcest.org/bigdata2019/?lang=zh>](http://www.ikcest.org/bigdata2019/?lang=zh)

### 思路

任务目标是利用遥感图像和访问数据对区域进行分类，因而设计为一个双输入单输出的网络。

### 环境

python 3.5; keras 2.1.6

### 代码结构

![](http://ww1.sinaimg.cn/large/006612YKgy1g3myatmph8j312o0hfjtc.jpg)

### 使用步骤

- 从链接中下载数据，放在origin_data文件夹中
- 运行preprocess/process_all.py , 进行数据的预处理
- 运行runs/main.py ，训练、评估、测试。
