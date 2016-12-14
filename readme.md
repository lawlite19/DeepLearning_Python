深度学习 Deep Learning
==============

- 有关神经网络的部分可以查看[这里](https://github.com/lawlite19/MachineLearning_Python)的`BP神经网络`的部分：[https://github.com/lawlite19/MachineLearning_Python](https://github.com/lawlite19/MachineLearning_Python)

## 一、CNN卷积神经网络
### 1、概述
- 典型的深度学习模型就是很深层的神经网络，包含多个`隐含层`，多隐层的神经网络很难直接使用`BP算法`进行直接训练，因为反向传播误差时往往会发散，很难收敛
- `CNN`节省训练开销的方式是**权共享weight sharing**，让**一组**神经元使用相同的权值
- 主要用于**图像识别**领域

### 2、卷积（convolution）特征提取
- `卷积核`（Convolution Kernel），也叫`过滤器filter`，由对应的权值`W`和偏置`b`体现
- 下图是`3x3`的卷积核在`5x5`的图像上做卷积的过程，就是矩阵做**点乘**之后的和
![enter description here][1]   
第`i`个隐含单元的输入就是：![$${W_{\rm{i}}}{x_{small}} + {b_i}$$](http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20%24%24%7BW_%7B%5Crm%7Bi%7D%7D%7D%7Bx_%7Bsmall%7D%7D%20&plus;%20%7Bb_i%7D%24%24)，其中![$${x_{small}}$$](http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20$${x_{small}}$$)就时与过滤器filter过滤到的图片
- 另外上图的步长`stride`为`1`，就是每个`filter`每次移动的距离
- 卷积特征提取的原理
 - 卷积特征提取利用了自然图像的统计平稳性，这一部分学习的特征也能用在另一部分上，所以对于这个图像上的所有位置，我们都能使用同样的学习特征。
 - 当有多个`filter`时，我们就可以学到多个特征，例如：轮廓、颜色等

- 多个过滤器`filter`（卷积核）
- 例子
![enter description here][2]


  [1]: ./images/CNN_01.gif "CNN_01.gif"
  [2]: ./images/CNN_02.gif "CNN_02.gif"