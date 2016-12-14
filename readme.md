深度学习 Deep Learning
==============
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
- 有关神经网络的部分可以查看这里的`BP神经网络`的部分：[https://github.com/lawlite19/MachineLearning_Python](https://github.com/lawlite19/MachineLearning_Python)

## 一、CNN卷积神经网络
### 1、概述
- 典型的深度学习模型就是很深层的神经网络，包含多个`隐含层`，多隐层的神经网络很难直接使用`BP算法`进行直接训练，因为反向传播误差时往往会发散，很难收敛
- `CNN`节省训练开销的方式是**权共享weight sharing**，让**一组**神经元使用相同的权值
- 主要用于**图像识别**领域

### 2、卷积（convolution）特征提取
- `卷积核`（Convolution Kernel），也叫`过滤器filter`，由对应的权值`W`和偏置`b`体现
- 下图是`3x3`的卷积核在`5x5`的图像上做卷积的过程，就是矩阵做**点乘**之后的和
![enter description here][1]
```mathjax!
\[{W_{\rm{i}}}{x_{small}} + {b_i}\]
```
```mathjax!
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)
```



  [1]: ./images/CNN_01.gif "CNN_01.gif"