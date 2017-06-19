深度学习 Deep Learning
==============

- 有关神经网络的部分可以查看[这里](https://github.com/lawlite19/MachineLearning_Python)的`BP神经网络`的部分：[https://github.com/lawlite19/MachineLearning_Python](https://github.com/lawlite19/MachineLearning_Python)

## 一、CNN卷积神经网络
- 参考文章：http://cs231n.github.io/convolutional-networks/#conv

### 1、概述
- 典型的深度学习模型就是很深层的神经网络，包含多个`隐含层`，多隐层的神经网络很难直接使用`BP算法`进行直接训练，因为反向传播误差时往往会发散，很难收敛
- `CNN`节省训练开销的方式是**权共享weight sharing**，让**一组**神经元使用相同的权值
- 主要用于**图像识别**领域

### 2、卷积（Convolution）特征提取
- `卷积核`（Convolution Kernel），也叫`过滤器filter`，由对应的权值`W`和偏置`b`体现
- 下图是`3x3`的卷积核在`5x5`的图像上做卷积的过程，就是矩阵做**点乘**之后的和
![enter description here][1]   
第`i`个隐含单元的输入就是：![$${W_{\rm{i}}}{x_{small}} + {b_i}$$](http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20%24%24%7BW_%7B%5Crm%7Bi%7D%7D%7D%7Bx_%7Bsmall%7D%7D%20&plus;%20%7Bb_i%7D%24%24)，其中![$${x_{small}}$$](http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20$${x_{small}}$$)就时与过滤器filter过滤到的图片
- 另外上图的步长`stride`为`1`，就是每个`filter`每次移动的距离
- 卷积特征提取的原理
 - 卷积特征提取利用了自然图像的统计平稳性，这一部分学习的特征也能用在另一部分上，所以对于这个图像上的所有位置，我们都能使用同样的学习特征。
 - 当有多个`filter`时，我们就可以学到多个特征，例如：轮廓、颜色等

- 多个过滤器`filter`（卷积核）
 - 例子如下
![enter description here][2]
 - 一张图片有`RGB`三个颜色通道，则对应的filter过滤器也是三维的，图像经过每个`filter`做卷积运算后都会得到对应提取特征的图像，途中两个`filter`:W0和W1,输出的就是**两个**图像
 - 这里的步长`stride`为`2`（一般就取2,3）
 - 在原图上添加`zero-padding`，它是`超参数`，主要用于控制输出的大小
 - 同样也是做卷积操作，以下图的一步卷积操作为例：   
 与w0[:,:,0]卷积：`0x(-1)+0x0+0x1+0x1+0x0+1x(-1)+1x0+1x(-1)+2x0=-2`   
 与w0[:,:,1]卷积：`2x1+1x(-1)+1x1=2`   
 与w0[:,:,2]卷积：`1x(-1)+1x(-1)=-2`   
 最终结果：`-2+2+(-2)+1=-1`  (1为偏置)
![enter description here][3]

### 3、池化（Pooling）
- 也叫做**下采样**
- `Pooling`过程
 - 把提取之后的特征看做一个矩阵，并在这个矩阵上划分出几个不重合的区域，
 - 然后在每个区域上计算该区域内特征的**均值**或**最大值**，然后用这些均值或最大值参与后续的训练
 ![enter description here][4]
 -下图是使用最大`Pooling`的方法之后的结果
 ![enter description here][5]
- `Pooling`的好处
 - 很明显就是减少参数
 - `Pooling`就有平移不变性（(translation invariant）
 如图`feature map`是`12x12`大小的图片，Pooling区域为6x6,所以池化后得到的`feature map`为`2x2`,假设白色像素值为1，灰色像素值为0，若采用`max pooling`之后，左上角窗口值为**1**
 ![enter description here][6]      
 将图像右移一个像素，左上角窗口值仍然为**1**     
 ![enter description here][7]     
 将图像缩放之后，左上角窗口值仍然为**1**     
 ![enter description here][8]    
- `Pooling`的方法中`average`方法对背景保留更好，`max`对纹理提取更好
- 深度学习可以进行多次卷积、池化操作

### 4、激活层
- 在每次卷积操作之后一般都会经过一个**非线性层**，也是**激活层**
- 现在一般选择是`ReLu`,层次越深，相对于其他的函数效果较好，还有`Sigmod,tanh`函数等
![enter description here][9]
- `sigmod`和`tanh`都存在**饱和**的问题，如上图所示，当x轴上的值较大时，对应的梯度几乎为0，若是利用**BP反向传播**算法， 可能造成梯度消失的情况，也就学不到东西了

### 5、全连接层 Fully connected layer
- 将多次卷积和池化后的图像展开进行全连接，如下图所示。
![enter description here][10]
- 接下来就可以通过**BP反向传播**进行训练了
- 所以总结起来，结构可以是这样的
![enter description here][11]

-------------------

### 6、CNN是如何工作的
- 看到知乎上的一个回答还不错：https://www.zhihu.com/question/52668301
- 每个过滤器可以被看成是特征标识符`（ feature identifiers）`
- 如下图一个曲线检测器对应的值     
![enter description here][12]
- 我们有一张图片，当过滤器移动到左上角时，进行**卷积运算**     
![enter description here][13]
- 当与我们的过滤器的形状很相似时，得到的值会很大
![enter description here][14]     
- 若是滑动到其他的部分，可以看出很不一样，对应的值就会很小，然后进行激活层的映射。     
![enter description here][15]
- 过滤器`filter`的值怎么求到，就是我们通过`BP`训练得到的。


--------------------

### 7、CNN的Tensorflow实现
- 代码和说明放到了另外一个`Tensorflow`学习的仓库里了
- [全部代码](https://github.com/lawlite19/MachineLearning_TensorFlow/blob/master/Mnist_03_CNN/mnist_cnn.py)：https://github.com/lawlite19/MachineLearning_TensorFlow/blob/master/Mnist_03_CNN/mnist_cnn.py
- 说明部分（第七部分）：https://github.com/lawlite19/MachineLearning_TensorFlow

--------------------------------

### 8、CNN公式推导
#### （1）说明
- 参考论文：http://cogprints.org/5869/1/cnn_tutorial.pdf
- 或者在这里查看：https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/cnn_tutorial.pdf
- **BP神经网络**之前写过推导，可以查看这里的第三部分BP神经网络：https://github.com/lawlite19/MachineLearning_Python
- 我们假设CNN中每个**卷积层**下面都跟着一个**Pooling池化层**（下采样层）
- 文章的理解可能会有问题

#### （2）符号说明
- `l`..................当前层
- ![$${{M_j}}$$](http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Clarge%20$${{M_j}}$$)..................输入maps的集合
- up()..................上采样函数
- ㅇ....................表示对应每个元素相乘
- `β`....................下采样对应的“权重”（定义为常量）
- ![$${p_i^{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bp_i%5E%7Bl%20-%201%7D%7D%24%24)...................![$${{\rm{x}}_i^{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Crm%7Bx%7D%7D_i%5E%7Bl%20-%201%7D%7D%24%24)中在卷积运算中逐个与![$${k_{ij}^l}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bk_%7Bij%7D%5El%7D%24%24)相乘的`patch`
- down()..................下采样函数



#### （3）卷积层
- 1）卷积层计算公式
 - ![$${\rm{x}}_j^l = f(\sum\limits_{i \in {M_j}} {{\rm{x}}_i^{l - 1}*k_{ij}^l + b_j^l} )$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%5Crm%7Bx%7D%7D_j%5El%20%3D%20f%28%5Csum%5Climits_%7Bi%20%5Cin%20%7BM_j%7D%7D%20%7B%7B%5Crm%7Bx%7D%7D_i%5E%7Bl%20-%201%7D*k_%7Bij%7D%5El%20&plus;%20b_j%5El%7D%20%29%24%24)
 - ![$${\rm{x}}_j^l$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%5Crm%7Bx%7D%7D_j%5El%24%24) 表示第`l`层的第`j`个`feature map`（**特征图**）
 - 可以对照到上面**多个卷积核**的例子看
 - `j`相当于是第几个卷积核
 - `i`相当于对应卷积核或是map的维度


- 2）卷积层梯度计算
 - paper中叫做使用**BP**计算当前层layer单元的**灵敏度**（sensitivity）
 - 也就是误差的计算，之前我在**BP神经网络**中推导过，这里不再给出
 - 当前层的第`j`个unit的灵敏度![$$\delta _{\rm{j}}^l$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5El%24%24)结果就是：先对下一层的节点（连接到当前层`l`的感兴趣节点的第`l+1`层的节点）的灵敏度求和（得到![$$\delta _{\rm{j}}^{l + 1}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5E%7Bl%20&plus;%201%7D%24%24)），然后乘以这些连接对应的权值（连接第`l`层感兴趣节点和第`l+1`层节点的权值）`W`。再乘以当前层`l`的该神经元节点的输入`u`的激活函数**f的导数值**
 - ![$$\delta _{\rm{j}}^l = \beta _j^{l + 1}({f^'}(u_j^l) \circ up(\delta _{\rm{j}}^{l + 1}))$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5El%20%3D%20%5Cbeta%20_j%5E%7Bl%20&plus;%201%7D%28%7Bf%5E%27%7D%28u_j%5El%29%20%5Ccirc%20up%28%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5E%7Bl%20&plus;%201%7D%29%29%24%24)
 - 下采样的“weights”可以定义为常量`β`（可以查看下面Pooling层输出的表示）
 - `up`表示**上采样**操作，因为我们之前假设每个卷积层之后跟着一个Pooling层，所以反向传播需要进行上采样
 - `up`上采样可以使用**克罗内克积**（Kronecker）实现，如果A是一个 `m x n` 的矩阵，而B是一个 `p x q` 的矩阵，克罗内克积则是一个 `mp x nq` 的矩阵，![$$up({\rm{x}}) = {\rm{x}} \otimes {1_{n \times n}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24up%28%7B%5Crm%7Bx%7D%7D%29%20%3D%20%7B%5Crm%7Bx%7D%7D%20%5Cotimes%20%7B1_%7Bn%20%5Ctimes%20n%7D%7D%24%24)
 ![enter description here][16]
 - 所以**偏置**的**梯度**为：![$${{\partial E} \over {\partial {b_j}}} = \sum\limits_{u,v} {{{(\delta _{\rm{j}}^l)}_{uv}}} $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Cpartial%20E%7D%20%5Cover%20%7B%5Cpartial%20%7Bb_j%7D%7D%7D%20%3D%20%5Csum%5Climits_%7Bu%2Cv%7D%20%7B%7B%7B%28%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5El%29%7D_%7Buv%7D%7D%7D%20%24%24) （因为神经网络中对`b`的梯度为：(![$${{\partial E} \over {\partial b}} = {{\partial E} \over {\partial u}}{{\partial u} \over {\partial b}} = \delta $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Cpartial%20E%7D%20%5Cover%20%7B%5Cpartial%20b%7D%7D%20%3D%20%7B%7B%5Cpartial%20E%7D%20%5Cover%20%7B%5Cpartial%20u%7D%7D%7B%7B%5Cpartial%20u%7D%20%5Cover%20%7B%5Cpartial%20b%7D%7D%20%3D%20%5Cdelta%20%24%24)（δ就是误差，根据定义的**代价函数E**得来的），其中`u`为layer的输入：![$${u^l} = {W^l}{{\rm{x}}^{l - 1}} + {b^l}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bu%5El%7D%20%3D%20%7BW%5El%7D%7B%7B%5Crm%7Bx%7D%7D%5E%7Bl%20-%201%7D%7D%20&plus;%20%7Bb%5El%7D%24%24)）
 - 所以**卷积核权值的梯度**为：![$${{\partial E} \over {\partial k_{ij}^l}} = \sum\limits_{u,v} {{{(\delta _{\rm{j}}^l)}_{uv}}(p_i^{l - 1})uv} $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Cpartial%20E%7D%20%5Cover%20%7B%5Cpartial%20k_%7Bij%7D%5El%7D%7D%20%3D%20%5Csum%5Climits_%7Bu%2Cv%7D%20%7B%7B%7B%28%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5El%29%7D_%7Buv%7D%7D%28p_i%5E%7Bl%20-%201%7D%29uv%7D%20%24%24) （其中：![$${p_i^{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bp_i%5E%7Bl%20-%201%7D%7D%24%24)为![$${{\rm{x}}_i^{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Crm%7Bx%7D%7D_i%5E%7Bl%20-%201%7D%7D%24%24)中在卷积运算中逐个与![$${k_{ij}^l}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bk_%7Bij%7D%5El%7D%24%24)相乘的`patch`，因为权重的系数就是对应的`patch`，对权重求导，就是这个系数）


#### （4）子采样层（Sub-sampling Layers）
- 1）子采样层计算公式
 - ![$${\rm{x}}_j^l = f(\beta _j^ldown({\rm{x}}_{\rm{j}}^{l - 1}) + b_j^l)$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%5Crm%7Bx%7D%7D_j%5El%20%3D%20f%28%5Cbeta%20_j%5Eldown%28%7B%5Crm%7Bx%7D%7D_%7B%5Crm%7Bj%7D%7D%5E%7Bl%20-%201%7D%29%20&plus;%20b_j%5El%29%24%24)
 - 乘以一个常数权重`β`，再加上偏置，然后再调用激活函数（这里和上面的pooling的操作有所不同，但总的来数还是下采样的过程）


- 2）梯度计算
 - 敏感度公式:![$$\delta _{\rm{j}}^l = {f^'}(u_j^l) \circ conv2(\delta _{\rm{j}}^{l + 1},rot180(k_j^{l + 1}),'full')$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5El%20%3D%20%7Bf%5E%27%7D%28u_j%5El%29%20%5Ccirc%20conv2%28%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5E%7Bl%20&plus;%201%7D%2Crot180%28k_j%5E%7Bl%20&plus;%201%7D%29%2C%27full%27%29%24%24)
 - 和上面的其实类似，就是换成下一层对应的权重`k`，rot180()是旋转180度，因为卷积的时候是将卷积核旋转180度之后然后在点乘求和的
 - 对偏置`b`的梯度与上面的一样
 - 对于乘法偏置（文中叫 multiplicative bias）`β`的梯度为：![$${{\partial E} \over {\partial {\beta _j}}} = \sum\limits_{u,v} {{{(\delta _{\rm{j}}^l \circ d_j^l)}_{uv}}} $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Cpartial%20E%7D%20%5Cover%20%7B%5Cpartial%20%7B%5Cbeta%20_j%7D%7D%7D%20%3D%20%5Csum%5Climits_%7Bu%2Cv%7D%20%7B%7B%7B%28%5Cdelta%20_%7B%5Crm%7Bj%7D%7D%5El%20%5Ccirc%20d_j%5El%29%7D_%7Buv%7D%7D%7D%20%24%24)，其中![$$d_j^l = down({\rm{x}}_j^{l - 1})$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24d_j%5El%20%3D%20down%28%7B%5Crm%7Bx%7D%7D_j%5E%7Bl%20-%201%7D%29%24%24)


## 二、权重初始化问题1_Sigmoid\tanh\Softsign激励函数
### 1、说明
- 参考论文：http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
- 或者查看[这里](https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/Understanding%20the%20difficulty%20of%20training%20deep%20feedforward%20neural%20networks.pdf)，我放在github上了：https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/Understanding%20the%20difficulty%20of%20training%20deep%20feedforward%20neural%20networks.pdf
- 这是2010年的论文，当时只是讨论的`Sigmoid`，`tanh`和`Softsign`激活函数，解决深层神经网络梯度消失的问题，并提出了一种初始化权重`weights`的方法，但是对于`ReLu`激活函数还是失效的，下一篇再讲。

### 2、实验
- 论文先是指出了`Sigmoid`激励函数是不适合作为**深度学习**的激励函数的，因为它的**均值**总是大于`0`的，导致**后面**隐含层hidden layer的神经元趋于**饱和**    
![enter description here][17]
- 构建了含有4个隐层的神经网络，激活函数为`Sigmoid`,观察每一层的激活值的均值和标准差的岁训练次数的变化情况，`layer1`表示**第一个**隐含层，一次类推。
- 初始化权重![$${W_{ij}} \sim U[ - {1 \over {\sqrt n }},{1 \over {\sqrt n }}]$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7BW_%7Bij%7D%7D%20%5Csim%20U%5B%20-%20%7B1%20%5Cover%20%7B%5Csqrt%20n%20%7D%7D%2C%7B1%20%5Cover%20%7B%5Csqrt%20n%20%7D%7D%5D%24%24),即服从**均匀分布**
- 如下图所示，实线表示**均值mean value**，**垂直的条**表示标准差。
 - 因为使用的均匀分布进行的初始化，所以前几层x的均值近似为**0**，所以对应`Sigmoid`函数的值就是**0.5**
 - 但是最后一层layer4的输出很快就饱和了（激活值趋于0）,训练到大100的时候才慢慢恢复正常
 - 作者给出当有**5**个隐含层的时候，最后一层始终处于饱和状态
 - 标准差反应的是数值的**波动**，可以看出后面才有了标准差的值     
![enter description here][18]
- 直观解释
 - 最后使用的是`softmax(b+Wh)`作为预测，刚开始训练的时候不能够很好的预测`y`的值，因此误差梯度会迫使`Wh`趋于`0`，所以会使`h`的值趋于`0`
 - `h`就是上一层的激活输出，所以对应的激活值很快降为`0`
- `tanh`激活函数是关于原点对称的，趋于**0**是没有问题的，因为梯度能够反向传回去。     
![enter description here][19]

### 3、初试化方法公式推导
- 首先**代价函数**使用的是**交叉熵代价函数**，相比对于**二次代价函数**会更好，看下对比就知道了，二次代价函数较为平坦，所以使用梯度下降会比较慢。（图中`W1`表示第一层的权重，`W2`表示第二层的权重）    
![enter description here][20]
- 以`tanh`激活函数为例 
- 符号说明
 - ![$${z^i}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bz%5Ei%7D%24%24)………………………………第i层的激活值向量
 - ![$${{\rm{s}}^i}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Crm%7Bs%7D%7D%5Ei%7D%24%24)………………………………第i+1层的输入
 - x…………………………………输入
 - ![$${{\rm{n}}_i}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Crm%7Bn%7D%7D_i%7D%24%24)………………………………..第i层神经元个数
 - W………………………………权重
 - Cost………………………………代价函数
- 所以第`i+1`层的输入：    
![$${{\rm{s}}^i} = {z^i}{W^i} + {b^i}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Crm%7Bs%7D%7D%5Ei%7D%20%3D%20%7Bz%5Ei%7D%7BW%5Ei%7D%20&plus;%20%7Bb%5Ei%7D%24%24)
- 激活之后的值表示：    
![$${z^{i + 1}} = f({s^i})$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bz%5E%7Bi%20&plus;%201%7D%7D%20%3D%20f%28%7Bs%5Ei%7D%29%24%24)，
- `❶`根据BP反向传播可以得到：    
![$${{\partial Cost} \over {\partial s_k^i}} = {f^'}(s_k^i)W_{k, \bullet }^{i + 1}{{\partial Cost} \over {\partial {s^{i + 1}}}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20s_k%5Ei%7D%7D%20%3D%20%7Bf%5E%27%7D%28s_k%5Ei%29W_%7Bk%2C%20%5Cbullet%20%7D%5E%7Bi%20&plus;%201%7D%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bs%5E%7Bi%20&plus;%201%7D%7D%7D%7D%24%24)
 - `❷`权重的偏导（梯度）就为：    
 ![$${{\partial Cost} \over {\partial w_{l,k}^i}} = z_l^i{{\partial Cost} \over {\partial s_k^i}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20w_%7Bl%2Ck%7D%5Ei%7D%7D%20%3D%20z_l%5Ei%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20s_k%5Ei%7D%7D%24%24)
 - 还是BP反向传播的推导，可以查看我之前给的BP反向传播的推导。
 - 它这里`W`是从`0`开始的，所以对应可能不太一致。
- `tanh`的导数为：     
![$${[\tanh (x)]^'} = 1 - {[\tanh (x)]^2}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%5B%5Ctanh%20%28x%29%5D%5E%27%7D%20%3D%201%20-%20%7B%5B%5Ctanh%20%28x%29%5D%5E2%7D%24%24)
 - 所以：![$${f^'}(0) = 1$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bf%5E%27%7D%280%29%20%3D%201%24%24)
 - 当![$$s_k^i$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24s_k%5Ei%24%24)的很小时，可以得到：![$${f^'}(s_k^i) \approx 1$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bf%5E%27%7D%28s_k%5Ei%29%20%5Capprox%201%24%24)
 - 这里实际上他是假设激励函数是线性的，下一篇论文中也有提到。
- 根据**方差**公式：       
![$$Var(x) = E({x^2}) - {E^2}(x)$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%28x%29%20%3D%20E%28%7Bx%5E2%7D%29%20-%20%7BE%5E2%7D%28x%29%24%24)
可以得到`❸`：       
![$$Var[{z^i}] = Var[x]\prod\limits_{j = 0}^{i - 1} {{n_j}Var[{W^j}]} $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%5B%7Bz%5Ei%7D%5D%20%3D%20Var%5Bx%5D%5Cprod%5Climits_%7Bj%20%3D%200%7D%5E%7Bi%20-%201%7D%20%7B%7Bn_j%7DVar%5B%7BW%5Ej%7D%5D%7D%20%24%24)，     
推导如下：
 - ![$$Var(s) = Var(\sum\limits_i^n {{w_i}{x_i}} ) = \sum\limits_i^n {Var({w_i}{x_i})} $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%28s%29%20%3D%20Var%28%5Csum%5Climits_i%5En%20%7B%7Bw_i%7D%7Bx_i%7D%7D%20%29%20%3D%20%5Csum%5Climits_i%5En%20%7BVar%28%7Bw_i%7D%7Bx_i%7D%29%7D%20%24%24)
 - ![enter description here][21]（式子太长，直接截图的，没用LaTex解析）
 - 因为输入的**均值为0**，可以得到：![$$E(w) = E(x) = 0$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24E%28w%29%20%3D%20E%28x%29%20%3D%200%24%24)
 - 所以：![$$Var(wx) = Var(w)Var(x)$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%28wx%29%20%3D%20Var%28w%29Var%28x%29%24%24)，代入上面即可。
- 因为之前![$${f^'}(s_k^i) \approx 1$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bf%5E%27%7D%28s_k%5Ei%29%20%5Capprox%201%24%24)，所以可以得到`❹`：        
![$$V{\rm{ar}}[{{\partial Cost} \over {\partial {s^i}}}] = Var[{{\partial Cost} \over {\partial {s^n}}}]\prod\limits_{j = i}^n {{n_{j + 1}}Var[{W^j}]} $$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24V%7B%5Crm%7Bar%7D%7D%5B%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bs%5Ei%7D%7D%7D%5D%20%3D%20Var%5B%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bs%5En%7D%7D%7D%5D%5Cprod%5Climits_%7Bj%20%3D%20i%7D%5En%20%7B%7Bn_%7Bj%20&plus;%201%7D%7DVar%5B%7BW%5Ej%7D%5D%7D%20%24%24)
- 将`❸❹`代入到对权重w偏导`❷`（即为梯度）的方差为:![$$Var[{{\partial Cost} \over {\partial {w^i}}}] = \prod\limits_{j = 0}^{i - 1} {{n_j}Var[{W^j}]} \prod\limits_{j = i}^{n - 1} {{n_{j + 1}}Var[{W^j}]}  * Var[x]Var[{{\partial Cost} \over {\partial {s^n}}}]$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%5B%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bw%5Ei%7D%7D%7D%5D%20%3D%20%5Cprod%5Climits_%7Bj%20%3D%200%7D%5E%7Bi%20-%201%7D%20%7B%7Bn_j%7DVar%5B%7BW%5Ej%7D%5D%7D%20%5Cprod%5Climits_%7Bj%20%3D%20i%7D%5E%7Bn%20-%201%7D%20%7B%7Bn_%7Bj%20&plus;%201%7D%7DVar%5B%7BW%5Ej%7D%5D%7D%20*%20Var%5Bx%5DVar%5B%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bs%5En%7D%7D%7D%5D%24%24)
- 对于正向传播，我们希望： ![$$\forall (i,j),Var[{z^i}] = Var[{z^j}]$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%5Cforall%20%28i%2Cj%29%2CVar%5B%7Bz%5Ei%7D%5D%20%3D%20Var%5B%7Bz%5Ej%7D%5D%24%24)
 - 对于反向传播，同样希望：![$$\forall (i,j),Var[{{\partial Cost} \over {\partial {s^i}}}] = Var[{{\partial Cost} \over {\partial {s^j}}}]$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%5Cforall%20%28i%2Cj%29%2CVar%5B%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bs%5Ei%7D%7D%7D%5D%20%3D%20Var%5B%7B%7B%5Cpartial%20Cost%7D%20%5Cover%20%7B%5Cpartial%20%7Bs%5Ej%7D%7D%7D%5D%24%24)
 - 两种情况可以转化为：               
 `❺`![$${n_i}Var[{w^{\rm{i}}}] = 1$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bn_i%7DVar%5B%7Bw%5E%7B%5Crm%7Bi%7D%7D%7D%5D%20%3D%201%24%24)和`❻`![$${n_{i + 1}}Var[{w^{\rm{i}}}] = 1$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bn_%7Bi%20&plus;%201%7D%7DVar%5B%7Bw%5E%7B%5Crm%7Bi%7D%7D%7D%5D%20%3D%201%24%24)      
（比如第一种：            
![](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%5B%7Bz%5Ei%7D%5D%20%3D%20Var%5Bx%5D%5Cprod%5Climits_%7Bj%20%3D%200%7D%5E%7Bi%20-%201%7D%20%7B%7Bn_j%7DVar%5B%7BW%5Ej%7D%5D%7D%20%24%24)，![$$V{\rm{ar}}({z^i}) = Var(x)$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24V%7B%5Crm%7Bar%7D%7D%28%7Bz%5Ei%7D%29%20%3D%20Var%28x%29%24%24) ，所以 ![$${n_i}Var[{w^{\rm{i}}}] = 1$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7Bn_i%7DVar%5B%7Bw%5E%7B%5Crm%7Bi%7D%7D%7D%5D%20%3D%201%24%24)，第二种情况同理）
- `❺❻`两式相加得：          
![$$Var[{W^i}] = {2 \over {{n_i} + {n_{i + 1}}}}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24Var%5B%7BW%5Ei%7D%5D%20%3D%20%7B2%20%5Cover%20%7B%7Bn_i%7D%20&plus;%20%7Bn_%7Bi%20&plus;%201%7D%7D%7D%7D%24%24)
- 最后提出了一个归一化的初始化方法,因为**W**服从**均匀分布**`U[-c,c]`，根据均匀分布的方差公式得：         
![$${{{{[c - ( - c)]}^2}} \over {12}} = {{{c^2}} \over 3}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B%7B%7B%7B%5Bc%20-%20%28%20-%20c%29%5D%7D%5E2%7D%7D%20%5Cover%20%7B12%7D%7D%20%3D%20%7B%7B%7Bc%5E2%7D%7D%20%5Cover%203%7D%24%24)
 - 所以：![$${2 \over {{n_i} + {n_{i + 1}}}} = {{{c^2}} \over 3}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24%7B2%20%5Cover%20%7B%7Bn_i%7D%20&plus;%20%7Bn_%7Bi%20&plus;%201%7D%7D%7D%7D%20%3D%20%7B%7B%7Bc%5E2%7D%7D%20%5Cover%203%7D%24%24)
 - 求出，![$$c = {{\sqrt 6 } \over {\sqrt {{n_i} + {n_{i + 1}}} }}$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24c%20%3D%20%7B%7B%5Csqrt%206%20%7D%20%5Cover%20%7B%5Csqrt%20%7B%7Bn_i%7D%20&plus;%20%7Bn_%7Bi%20&plus;%201%7D%7D%7D%20%7D%7D%24%24)
 - 所以**◆◆**最终给出初始化权重的方法为：          
 ![$$W \sim U[ - {{\sqrt 6 } \over {\sqrt {{n_i} + {n_{i + 1}}} }},{{\sqrt 6 } \over {\sqrt {{n_i} + {n_{i + 1}}} }}]$$](http://latex.codecogs.com/gif.latex?%5Clarge%20%24%24W%20%5Csim%20U%5B%20-%20%7B%7B%5Csqrt%206%20%7D%20%5Cover%20%7B%5Csqrt%20%7B%7Bn_i%7D%20&plus;%20%7Bn_%7Bi%20&plus;%201%7D%7D%7D%20%7D%7D%2C%7B%7B%5Csqrt%206%20%7D%20%5Cover%20%7B%5Csqrt%20%7B%7Bn_i%7D%20&plus;%20%7Bn_%7Bi%20&plus;%201%7D%7D%7D%20%7D%7D%5D%24%24)
- 这就是**Xavier初始化**方法

------------------------------------------------------

## 三、权重初始化问题2_`ReLu`激励函数
### 1、说明
- 参考论文：https://arxiv.org/pdf/1502.01852v1.pdf
- 或者查看[这里](https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/Delving%20Deep%20into%20Rectifiers%20Surpassing%20Human-Level%20Performance%20on%20ImageNet%20Classification%EF%BC%88S%E5%9E%8B%E5%87%BD%E6%95%B0%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%EF%BC%89.pdf)，我放在github上了：https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/Delving%20Deep%20into%20Rectifiers%20Surpassing%20Human-Level%20Performance%20on%20ImageNet%20Classification%EF%BC%88S%E5%9E%8B%E5%87%BD%E6%95%B0%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%EF%BC%89.pdf


### 2、`ReLu/PReLu`激励函数
- 目前`ReLu`激活函数使用比较多，而上面一篇论文没有讨论，如果还是使用同样初始化权重的方法（**Xavier初始化**）会有问题
- PReLu函数定义如下：
 - ![enter description here][22]
 - 等价于：![$$f({y_i}) = \max (0,{y_i}) + {a_i}\min (0,{y_i})$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24f%28%7By_i%7D%29%20%3D%20%5Cmax%20%280%2C%7By_i%7D%29%20&plus;%20%7Ba_i%7D%5Cmin%20%280%2C%7By_i%7D%29%24%24)
- ReLu（左）和PReLu（右）激活函数图像             
![enter description here][23]

### 3、前向传播推导
- 符号说明
 - ε……………………………………目标函数
 - μ……………………………………动量
 - α……………………………………学习率
 - f()………………………………激励函数
 - l……………………………………当前层
 - L……………………………………神经网络总层数
 - k……………………………………过滤器`filter`的大小
 - c……………………………………输入通道个数
 - x……………………………………`k^2c*1`的向量
 - d……………………………………过滤器`filter`的个数
 - b……………………………………偏置向量
- ![$${y_l} = {W_l}{{\rm{x}}_l} + {b_l}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7By_l%7D%20%3D%20%7BW_l%7D%7B%7B%5Crm%7Bx%7D%7D_l%7D%20&plus;%20%7Bb_l%7D%24%24)..............................................................(1)
 - ![$${{\rm{x}}_l} = f({y_{l - 1}})$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B%7B%5Crm%7Bx%7D%7D_l%7D%20%3D%20f%28%7By_%7Bl%20-%201%7D%7D%29%24%24) 
 - ![$${{\rm{c}}_l} = {d_{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B%7B%5Crm%7Bc%7D%7D_l%7D%20%3D%20%7Bd_%7Bl%20-%201%7D%7D%24%24)
- 根据式`(1)`得：     
![$$Var[{y_l}] = {n_l}Var[{w_l}{x_l}]$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24Var%5B%7By_l%7D%5D%20%3D%20%7Bn_l%7DVar%5B%7Bw_l%7D%7Bx_l%7D%5D%24%24)..................................................(2)
- 因为初始化权重`w`均值为0，所以**期望**：![$$E({w_l}) = 0$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24E%28%7Bw_l%7D%29%20%3D%200%24%24)，**方差**：![$$Var[{w_l}] = E(w_l^2) - {E^2}({w_l}) = E(w_l^2)$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24Var%5B%7Bw_l%7D%5D%20%3D%20E%28w_l%5E2%29%20-%20%7BE%5E2%7D%28%7Bw_l%7D%29%20%3D%20E%28w_l%5E2%29%24%24)
- 根据式`(2)`继续推导：     
![enter description here][24]............................................(3)
 - 对于`x`来说：![$$Var[{x_l}] \ne E[x_l^2]$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24Var%5B%7Bx_l%7D%5D%20%5Cne%20E%5Bx_l%5E2%5D%24%24)，除非`x`的均值也是0,
 - 对于`ReLu`函数来说：![$${x_l} = \max (0,{y_{l - 1}})$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7Bx_l%7D%20%3D%20%5Cmax%20%280%2C%7By_%7Bl%20-%201%7D%7D%29%24%24)，所以不可能均值为0
- `w`满足对称区间的分布，并且偏置![$${b_{l - 1}} = 0$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7Bb_%7Bl%20-%201%7D%7D%20%3D%200%24%24)，所以![$${y_{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7By_%7Bl%20-%201%7D%7D%24%24)也满足对称区间的分布，所以：    
![enter description here][25]..........................................(4)
- 将上式`(4)`代入`(3)`中得：      
![$$Var[{y_l}] = {1 \over 2}{n_l}Var[{w_l}]Var[{y_{l - 1}}]$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24Var%5B%7By_l%7D%5D%20%3D%20%7B1%20%5Cover%202%7D%7Bn_l%7DVar%5B%7Bw_l%7D%5DVar%5B%7By_%7Bl%20-%201%7D%7D%5D%24%24).......................................................(5)
- 所以对于`L`层:            
![$$Var[{y_L}] = Var[{y_1}]\prod\limits_{l = 2}^L {{1 \over 2}{n_l}Var[{w_l}]} $$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24Var%5B%7By_L%7D%5D%20%3D%20Var%5B%7By_1%7D%5D%5Cprod%5Climits_%7Bl%20%3D%202%7D%5EL%20%7B%7B1%20%5Cover%202%7D%7Bn_l%7DVar%5B%7Bw_l%7D%5D%7D%20%24%24).....................................................................(6)
 - 从上式可以看出，因为**累乘**的存在，若是![$${1 \over 2}{n_l}Var[{w_l}] < 1$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B1%20%5Cover%202%7D%7Bn_l%7DVar%5B%7Bw_l%7D%5D%20%3C%201%24%24)，每次累乘都会使方差缩小，若是大于`1`，每次会使方差当大。
 - 所以我们希望：     
 ![$${1 \over 2}{n_l}Var[{w_l}] = 1$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B1%20%5Cover%202%7D%7Bn_l%7DVar%5B%7Bw_l%7D%5D%20%3D%201%24%24)
- 所以**初始化方法**为：是`w`满足**均值为0**，**标准差**为![$$\sqrt {{2 \over {{n_l}}}} $$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5Csqrt%20%7B%7B2%20%5Cover%20%7B%7Bn_l%7D%7D%7D%7D%20%24%24)的**高斯分布**，同时**偏置**初始化为`0`


### 4、反向传播推导
- ![$$\Delta {{\rm{x}}_l} = \widehat {{W_l}}\Delta {y_l}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20%7B%7B%5Crm%7Bx%7D%7D_l%7D%20%3D%20%5Cwidehat%20%7B%7BW_l%7D%7D%5CDelta%20%7By_l%7D%24%24)....................................................(7)
 - 假设![$$\widehat {{W_l}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5Cwidehat%20%7B%7BW_l%7D%7D%24%24%24%24)和![$$\Delta {y_l}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20%7By_l%7D%24%24)相互独立的
 - 当![$$\widehat {{W_l}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5Cwidehat%20%7B%7BW_l%7D%7D%24%24%24%24)初始化Wie对称区间的分布时，可以得到：![$$\Delta {{\rm{x}}_l}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20%7B%7B%5Crm%7Bx%7D%7D_l%7D%24%24)的**均值**为0
 - `△x,△y`都表示梯度，即：   
 ![$$\Delta {\rm{x}} = {{\partial \varepsilon } \over {\partial {\rm{x}}}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20%7B%5Crm%7Bx%7D%7D%20%3D%20%7B%7B%5Cpartial%20%5Cvarepsilon%20%7D%20%5Cover%20%7B%5Cpartial%20%7B%5Crm%7Bx%7D%7D%7D%7D%24%24)，![$$\Delta y = {{\partial \varepsilon } \over {\partial y}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20y%20%3D%20%7B%7B%5Cpartial%20%5Cvarepsilon%20%7D%20%5Cover%20%7B%5Cpartial%20y%7D%7D%24%24)
- 根据**反向传播**：     
![$$\Delta {y_l} = {f^'}({y_l})\Delta {x_{l + 1}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20%7By_l%7D%20%3D%20%7Bf%5E%27%7D%28%7By_l%7D%29%5CDelta%20%7Bx_%7Bl%20&plus;%201%7D%7D%24%24)
 - 对于`ReLu`函数，**f的导数**为`0`或`1`，且**概率是相等的**，假设![$${f^'}({y_l})$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7Bf%5E%27%7D%28%7By_l%7D%29%24%24)和![$$\Delta {x_{l + 1}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5CDelta%20%7Bx_%7Bl%20&plus;%201%7D%7D%24%24)是相互独立的，
 - 所以：![$$E[\Delta {y_l}] = E[\Delta {x_{l + 1}}]/2 = 0$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24E%5B%5CDelta%20%7By_l%7D%5D%20%3D%20E%5B%5CDelta%20%7Bx_%7Bl%20&plus;%201%7D%7D%5D/2%20%3D%200%24%24)
- 所以：![$$E[{(\Delta {y_l})^2}] = Var[\Delta {y_l}] = {1 \over 2}Var[\Delta {x_{l + 1}}]$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24E%5B%7B%28%5CDelta%20%7By_l%7D%29%5E2%7D%5D%20%3D%20Var%5B%5CDelta%20%7By_l%7D%5D%20%3D%20%7B1%20%5Cover%202%7DVar%5B%5CDelta%20%7Bx_%7Bl%20&plus;%201%7D%7D%5D%24%24)...................................................(8)
- 根据`(7)`可以得到：              
![enter description here][26]
- 将`L`层展开得：        
![$$Var[\Delta {x_2}] = Var[\Delta {x_{L + 1}}]\prod\limits_{l = 2}^L {{1 \over 2}\widehat {{n_l}}Var[{w_l}]} $$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24Var%5B%5CDelta%20%7Bx_2%7D%5D%20%3D%20Var%5B%5CDelta%20%7Bx_%7BL%20&plus;%201%7D%7D%5D%5Cprod%5Climits_%7Bl%20%3D%202%7D%5EL%20%7B%7B1%20%5Cover%202%7D%5Cwidehat%20%7B%7Bn_l%7D%7DVar%5B%7Bw_l%7D%5D%7D%20%24%24)...........................................................(9)
- 同样令：![$${1 \over 2}\widehat {{n_l}}Var[{w_l}] = 1$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B1%20%5Cover%202%7D%5Cwidehat%20%7B%7Bn_l%7D%7DVar%5B%7Bw_l%7D%5D%20%3D%201%24%24)
 - 注意这里：![$$\widehat {{n_l}} = k_l^2{d_l}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5Cwidehat%20%7B%7Bn_l%7D%7D%20%3D%20k_l%5E2%7Bd_l%7D%24%24)，而![$${n_l} = k_l^2{c_l} = k_l^2{d_{l - 1}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7Bn_l%7D%20%3D%20k_l%5E2%7Bc_l%7D%20%3D%20k_l%5E2%7Bd_%7Bl%20-%201%7D%7D%24%24)

- 所以![$${{\rm{w}}_l}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B%7B%5Crm%7Bw%7D%7D_l%7D%24%24)应满足**均值为0**，**标准差**为：![$$\sqrt {{2 \over {\widehat {{n_l}}}}} $$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5Csqrt%20%7B%7B2%20%5Cover%20%7B%5Cwidehat%20%7B%7Bn_l%7D%7D%7D%7D%7D%20%24%24)的分布

### 5、正向和反向传播讨论、实验和**PReLu**函数
- 对于**正向和反向**两种初始化权重的方式都是可以的，论文中的模型都能够**收敛**
- 比如利用**反向传播**得到的初始化得到：![$$\prod\limits_{l = 2}^L {{1 \over 2}\widehat {{n_l}}Var[{w_l}]}  = 1$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%5Cprod%5Climits_%7Bl%20%3D%202%7D%5EL%20%7B%7B1%20%5Cover%202%7D%5Cwidehat%20%7B%7Bn_l%7D%7DVar%5B%7Bw_l%7D%5D%7D%20%3D%201%24%24)
- 对应到**正向传播**中得到：    
![enter description here][27]

- 所以也不是逐渐缩小的
- 实验给出了与**第一篇论文**的比较，如下图所示，当神经网络有30层时，**Xavier初始化权重**的方法（第一篇论文中的方法）已经不能收敛。     
![enter description here][28]
- 对于**PReLu激励函数**可以得到：![$${1 \over 2}(1 + {a^2}){n_l}Var[{w_l}] = 1$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B1%20%5Cover%202%7D%281%20&plus;%20%7Ba%5E2%7D%29%7Bn_l%7DVar%5B%7Bw_l%7D%5D%20%3D%201%24%24)
 - 当`a=0`时就是对应的**ReLu激励函数**
 - 当`a=1`是就是对应**线性函数**


---------------------------------------------------------------------

## 四、Batch Normalization（BN）批标准化
### 1、说明
- 参考论文：http://jmlr.org/proceedings/papers/v37/ioffe15.pdf
- 或者查看[这里](https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/%EF%BC%88BN%EF%BC%89Batch%20Normalization%20Accelerating%20Deep%20Network%20Training%20by%20Reducing%20Internal%20Covariate%20Shift.pdf)，我放在github上了：https://github.com/lawlite19/DeepLearning_Python/blob/master/paper/%EF%BC%88BN%EF%BC%89Batch%20Normalization%20Accelerating%20Deep%20Network%20Training%20by%20Reducing%20Internal%20Covariate%20Shift.pdf

### 2、论文概述
- 2015年Google提出的Batch Normalization
- 训练深层的神经网络很复杂，因为训练时每一层输入的分布在变化，导致训练过程中的饱和，称这种现象为：`internal covariate shift`
- 需要降低**学习率Learning Rate**和注意**参数的初始化**
- 论文中提出的方法是对于每一个小的训练batch都进行**标准化（正态化）**
 - 允许使用较大的学习率
 - 不必太关心初始化的问题
 - 同时一些例子中不需要使用`Dropout`方法避免过拟合
 - 此方法在`ImageNet classification`比赛中获得`4.82% top-5`的测试错误率

### 3、`BN`思路
- 如果输入数据是**白化**的（whitened），网络会更快的**收敛**
 - 白化**目的**是降低数据的冗余性和特征的相关性，例如通过**线性变换**使数据为**0均值**和**单位方差**

- **并非直接标准化每一层**那么简单，如果不考虑归一化的影响，可能会降低梯度下降的影响
- 标准化与某个样本和所有样本都有关系
 - 解决上面的问题，我们希望对于任何参数值，都要满足想要的分布；
 - ![$$\widehat x{\rm{ = }}N{\rm{orm}}({\rm{x}},\chi )$$](http://latex.codecogs.com/gif.latex?%24%24%5Cwidehat%20x%7B%5Crm%7B%20%3D%20%7D%7DN%7B%5Crm%7Borm%7D%7D%28%7B%5Crm%7Bx%7D%7D%2C%5Cchi%20%29%24%24)
 - 对于反向传播，需要计算:![$${{\partial N{\rm{orm}}({\rm{x}},\chi )} \over {\partial {\rm{x}}}}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20N%7B%5Crm%7Borm%7D%7D%28%7B%5Crm%7Bx%7D%7D%2C%5Cchi%20%29%7D%20%5Cover%20%7B%5Cpartial%20%7B%5Crm%7Bx%7D%7D%7D%7D%24%24)和![$${{\partial N{\rm{orm}}({\rm{x}},\chi )} \over {\partial \chi }}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20N%7B%5Crm%7Borm%7D%7D%28%7B%5Crm%7Bx%7D%7D%2C%5Cchi%20%29%7D%20%5Cover%20%7B%5Cpartial%20%5Cchi%20%7D%7D%24%24)
 - 这样做的**计算代价**是非常大的，因为需要计算x的**协方差矩阵** 
 - 然后**白化**操作：![$${{{\rm{x}} - E[{\rm{x}}]} \over {\sqrt {Cov[{\rm{x}}]} }}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%7B%5Crm%7Bx%7D%7D%20-%20E%5B%7B%5Crm%7Bx%7D%7D%5D%7D%20%5Cover%20%7B%5Csqrt%20%7BCov%5B%7B%5Crm%7Bx%7D%7D%5D%7D%20%7D%7D%24%24)
- 上面两种都不行或是不好，进而得到了**BN**的方法
- 既然**白化每一层**的**输入代价非常大**，我们可以进行简化
- 简化1
 - 标准化特征的**每一个维度**而不是去标准化**所有的特征**，这样就不用求**协方差矩阵**了
 - 例如`d`维的输入：![$${\rm{x}} = ({x^{(1)}},{x^{(2)}}, \cdots ,{x^{(d)}})$$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Crm%7Bx%7D%7D%20%3D%20%28%7Bx%5E%7B%281%29%7D%7D%2C%7Bx%5E%7B%282%29%7D%7D%2C%20%5Ccdots%20%2C%7Bx%5E%7B%28d%29%7D%7D%29%24%24)
 - 标准化操作：           
 ![$${\widehat x^{(k)}} = {{{x^{(k)}} - E[{x^{(k)}}]} \over {\sqrt {Var[{{\rm{x}}^{(k)}}]} }}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Cwidehat%20x%5E%7B%28k%29%7D%7D%20%3D%20%7B%7B%7Bx%5E%7B%28k%29%7D%7D%20-%20E%5B%7Bx%5E%7B%28k%29%7D%7D%5D%7D%20%5Cover%20%7B%5Csqrt%20%7BVar%5B%7B%7B%5Crm%7Bx%7D%7D%5E%7B%28k%29%7D%7D%5D%7D%20%7D%7D%24%24)
 - 需要注意的是标准化操作可能会**降低数据的表达能力**,例如我们之前提到的**Sigmoid函数**：              
 ![enter description here][29]
 - 标准化之后**均值为0**，**方差为1**，数据就会落在**近似线性**的函数区域内，这样激活函数的意义就不明显
 - 所以对于每个 ，对应一对**参数**：![$${\gamma ^{(k)}},{\beta ^{(k)}}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Cgamma%20%5E%7B%28k%29%7D%7D%2C%7B%5Cbeta%20%5E%7B%28k%29%7D%7D%24%24) ，然后令：![$${y^{(k)}} = {\gamma ^{(k)}}{\widehat x^{(k)}} + {\beta ^{(k)}}$$](http://latex.codecogs.com/gif.latex?%24%24%7By%5E%7B%28k%29%7D%7D%20%3D%20%7B%5Cgamma%20%5E%7B%28k%29%7D%7D%7B%5Cwidehat%20x%5E%7B%28k%29%7D%7D%20&plus;%20%7B%5Cbeta%20%5E%7B%28k%29%7D%7D%24%24)
 - 从式子来看就是对标准化的数据进行**缩放和平移**，不至于使数据落在线性区域内，增加数据的表达能力（式子中如果：![$${\gamma ^{(k)}} = \sqrt {Var[{{\rm{x}}^{(k)}}]} $$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Cgamma%20%5E%7B%28k%29%7D%7D%20%3D%20%5Csqrt%20%7BVar%5B%7B%7B%5Crm%7Bx%7D%7D%5E%7B%28k%29%7D%7D%5D%7D%20%24%24)，![$${\beta ^{(k)}} = E[{x^{(k)}}]$$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Cbeta%20%5E%7B%28k%29%7D%7D%20%3D%20E%5B%7Bx%5E%7B%28k%29%7D%7D%5D%24%24) ，就会使**恢复到原来的值**了）
 - 但是这里还是使用的**全部的数据集**，但是如果使用**随机梯度下降**，可以选取一个**batch**进行训练
- 简化2
 - 第二种简化就是使用`mini-batch`进行`随机梯度下降`
 - 注意这里使用`mini-batch`也是标准化**每一个维度**上的特征，而不是所有的特征一起，因为若果`mini-batch`中的数据量小于特征的维度时，会产生**奇异协方差矩阵**， 对应的**行列式**的值为0，非满秩
 - 假设mini-batch 大小为`m`的`B`
 - ![$$B = \{ {x_{1 \ldots m}}\} $$](http://latex.codecogs.com/gif.latex?%24%24B%20%3D%20%5C%7B%20%7Bx_%7B1%20%5Cldots%20m%7D%7D%5C%7D%20%24%24)，对应的变换操作为：![$$B{N_{\gamma ,\beta }}:{x_{1 \ldots m}} \to {y_{1 \ldots m}}$$](http://latex.codecogs.com/gif.latex?%24%24B%7BN_%7B%5Cgamma%20%2C%5Cbeta%20%7D%7D%3A%7Bx_%7B1%20%5Cldots%20m%7D%7D%20%5Cto%20%7By_%7B1%20%5Cldots%20m%7D%7D%24%24)
 - 作者给出的批标准化的算法如下：               
 ![enter description here][30]
 - 算法中的`ε`是一个**常量**，为了保证数值的稳定性

- 反向传播求梯度：
 - 因为：![$${y^{(k)}} = {\gamma ^{(k)}}{\widehat x^{(k)}} + {\beta ^{(k)}}$$](http://latex.codecogs.com/gif.latex?%24%24%7By%5E%7B%28k%29%7D%7D%20%3D%20%7B%5Cgamma%20%5E%7B%28k%29%7D%7D%7B%5Cwidehat%20x%5E%7B%28k%29%7D%7D%20&plus;%20%7B%5Cbeta%20%5E%7B%28k%29%7D%7D%24%24)
 - 所以：![$${{\partial l} \over {\partial {{\widehat x}_i}}} = {{\partial l} \over {\partial {y_i}}}\gamma $$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7B%7B%5Cwidehat%20x%7D_i%7D%7D%7D%20%3D%20%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7By_i%7D%7D%7D%5Cgamma%20%24%24)
 - 因为：![$${\widehat x_i} = {{{x_i} - {\mu _B}} \over {\sqrt {\sigma _B^2 + \varepsilon } }}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Cwidehat%20x_i%7D%20%3D%20%7B%7B%7Bx_i%7D%20-%20%7B%5Cmu%20_B%7D%7D%20%5Cover%20%7B%5Csqrt%20%7B%5Csigma%20_B%5E2%20&plus;%20%5Cvarepsilon%20%7D%20%7D%7D%24%24)
 - 所以：![$${{\partial l} \over {\partial \sigma _B^2}} = {\sum\limits_{i = 1}^m {{{\partial l} \over {\partial {{\widehat x}_i}}}({x_i} - {u_B}){{ - 1} \over 2}(\sigma _B^2 + \varepsilon )} ^{ - {3 \over 2}}}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%5Csigma%20_B%5E2%7D%7D%20%3D%20%7B%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7B%7B%5Cwidehat%20x%7D_i%7D%7D%7D%28%7Bx_i%7D%20-%20%7Bu_B%7D%29%7B%7B%20-%201%7D%20%5Cover%202%7D%28%5Csigma%20_B%5E2%20&plus;%20%5Cvarepsilon%20%29%7D%20%5E%7B%20-%20%7B3%20%5Cover%202%7D%7D%7D%24%24)
 - ![$${{\partial l} \over {\partial {u_B}}} = \sum\limits_{{\rm{i = 1}}}^m {{{\partial l} \over {\partial {{\widehat x}_i}}}} {{ - 1} \over {\sqrt {\sigma _B^2 + \varepsilon } }}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7Bu_B%7D%7D%7D%20%3D%20%5Csum%5Climits_%7B%7B%5Crm%7Bi%20%3D%201%7D%7D%7D%5Em%20%7B%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7B%7B%5Cwidehat%20x%7D_i%7D%7D%7D%7D%20%7B%7B%20-%201%7D%20%5Cover%20%7B%5Csqrt%20%7B%5Csigma%20_B%5E2%20&plus;%20%5Cvarepsilon%20%7D%20%7D%7D%24%24)
 - 因为：![$${\mu _B} = {1 \over m}\sum\limits_{i = 1}^m {{x_i}} $$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Cmu%20_B%7D%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7Bx_i%7D%7D%20%24%24)和![$$\sigma _B^2 = {1 \over m}\sum\limits_{i = 1}^m {({x_i}}  - {\mu _B}{)^2}$$](http://latex.codecogs.com/gif.latex?%24%24%5Csigma%20_B%5E2%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%28%7Bx_i%7D%7D%20-%20%7B%5Cmu%20_B%7D%7B%29%5E2%7D%24%24)
 - 所以：![$${{\partial l} \over {\partial {x_i}}} = {{\partial l} \over {\partial {{\widehat x}_i}}}{1 \over {\sqrt {\sigma _B^2 + \varepsilon } }} + {{\partial l} \over {\partial \sigma _B^2}}{{2({x_i} - {\mu _B})} \over m} + {{\partial l} \over {\partial {u_B}}}{1 \over m}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7Bx_i%7D%7D%7D%20%3D%20%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7B%7B%5Cwidehat%20x%7D_i%7D%7D%7D%7B1%20%5Cover%20%7B%5Csqrt%20%7B%5Csigma%20_B%5E2%20&plus;%20%5Cvarepsilon%20%7D%20%7D%7D%20&plus;%20%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%5Csigma%20_B%5E2%7D%7D%7B%7B2%28%7Bx_i%7D%20-%20%7B%5Cmu%20_B%7D%29%7D%20%5Cover%20m%7D%20&plus;%20%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7Bu_B%7D%7D%7D%7B1%20%5Cover%20m%7D%24%24)
 - 所以：![$${{\partial l} \over {\partial \gamma }} = \sum\limits_{i = 1}^m {{{\partial l} \over {\partial {y_i}}}} {\widehat x_i}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%5Cgamma%20%7D%7D%20%3D%20%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7By_i%7D%7D%7D%7D%20%7B%5Cwidehat%20x_i%7D%24%24)
 - ![$${{\partial l} \over {\partial \beta }} = \sum\limits_{i = 1}^m {{{\partial l} \over {\partial {y_i}}}} $$](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%5Cbeta%20%7D%7D%20%3D%20%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%5Cpartial%20l%7D%20%5Cover%20%7B%5Cpartial%20%7By_i%7D%7D%7D%7D%20%24%24)
- 对于**BN变换**是**可微分**的，随着网络的训练，网络层可以持续学到输入的分布。

### 4、`BN`网络的训练和推断
- 按照BN方法，输入数据`x`会经过变化得到`BN（x）`，然后可以通过**随机梯度下降**进行训练，标准化是在mini-batch上所以是非常高效的。
- 但是对于推断我们希望输出只取决于输入，而对于输入**只有一个实例数据**，无法得到`mini-batch`的其他实例，就**无法求对应的均值和方差**了。
- 可以通过从所有**训练实例中获得的统计量**来**代替**mini-batch中m个训练实例获得统计量均值和方差
- 我们对每个`mini-batch`做标准化，可以对记住每个`mini-batch`的B，然后得到**全局统计量**
- ![$$E[x] \leftarrow {E_B}[{\mu _B}]$$](http://latex.codecogs.com/gif.latex?%24%24E%5Bx%5D%20%5Cleftarrow%20%7BE_B%7D%5B%7B%5Cmu%20_B%7D%5D%24%24)
- ![$$Var[x] \leftarrow {m \over {m - 1}}{E_B}[\sigma _B^2]$$](http://latex.codecogs.com/gif.latex?%24%24Var%5Bx%5D%20%5Cleftarrow%20%7Bm%20%5Cover%20%7Bm%20-%201%7D%7D%7BE_B%7D%5B%5Csigma%20_B%5E2%5D%24%24)（这里方差采用的是**无偏**方差估计）
- 所以**推断**采用`BN`的方式为：
 - ![$$\eqalign{
  & y = \gamma {{x - E(x)} \over {\sqrt {Var[x] + \varepsilon } }} + \beta   \cr 
  & {\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt} {\kern 1pt}  = {\gamma  \over {\sqrt {Var[x] + \varepsilon } }}x + (\beta  - {{\gamma E[x]} \over {\sqrt {Var[x] + \varepsilon } }}) \cr} $$](http://latex.codecogs.com/gif.latex?%24%24%5Ceqalign%7B%20%26%20y%20%3D%20%5Cgamma%20%7B%7Bx%20-%20E%28x%29%7D%20%5Cover%20%7B%5Csqrt%20%7BVar%5Bx%5D%20&plus;%20%5Cvarepsilon%20%7D%20%7D%7D%20&plus;%20%5Cbeta%20%5Ccr%20%26%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%7B%5Ckern%201pt%7D%20%3D%20%7B%5Cgamma%20%5Cover%20%7B%5Csqrt%20%7BVar%5Bx%5D%20&plus;%20%5Cvarepsilon%20%7D%20%7D%7Dx%20&plus;%20%28%5Cbeta%20-%20%7B%7B%5Cgamma%20E%5Bx%5D%7D%20%5Cover%20%7B%5Csqrt%20%7BVar%5Bx%5D%20&plus;%20%5Cvarepsilon%20%7D%20%7D%7D%29%20%5Ccr%7D%20%24%24)
- 作者给出的完整算法：                
![enter description here][31]

### 5、实验
- 最后给出的实验可以看出使用BN的方式训练**精准度**很高而且很**稳定**。
![enter description here][32]

---------------------------------------------------------------------


## 五、RNN和LSTM_01基础
- 由于 `github` 中不支持 `Mathjax` 公式，查看请移步[我的博客](http://lawlite.me/2017/06/14/RNN-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-01/)

- Tensorflow中实现一个例子：[查看这里](http://lawlite.me/2017/06/16/RNN-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-02Tensorflow%E4%B8%AD%E7%9A%84%E5%AE%9E%E7%8E%B0/)













 


  [1]: ./images/CNN_01.gif "CNN_01.gif"
  [2]: ./images/CNN_02.gif "CNN_02.gif"
  [3]: ./images/CNN_03.png "CNN_03.png"
  [4]: ./images/CNN_04.gif "CNN_04.gif"
  [5]: ./images/CNN_05.png "CNN_05.png"
  [6]: ./images/CNN_06.png "CNN_06.png"
  [7]: ./images/CNN_07.png "CNN_07.png"
  [8]: ./images/CNN_08.png "CNN_08.png"
  [9]: ./images/CNN_10.png "CNN_10.png"
  [10]: ./images/CNN_09.png "CNN_09.png"
  [11]: ./images/CNN_11.png "CNN_11.png"
  [12]: ./images/CNN_12.png "CNN_12.png"
  [13]: ./images/CNN_13.png "CNN_13.png"
  [14]: ./images/CNN_15.png "CNN_15.png"
  [15]: ./images/CNN_14.png "CNN_14.png"
  [16]: ./images/CNN_16.jpg "CNN_16.jpg"
  [17]: ./images/Weights_initialization_01.png "Weights_initialization_01.png"
  [18]: ./images/Weights_initialization_02.png "Weights_initialization_02.png"
  [19]: ./images/Weights_initialization_03.png "Weights_initialization_03.png"
  [20]: ./images/Weights_initialization_04.png "Weights_initialization_04.png"
  [21]: ./images/Weights_initialization_05.png "Weights_initialization_05.png"
  [22]: ./images/Weights_initialization_06.png "Weights_initialization_06.png"
  [23]: ./images/Weights_initialization_07.png "Weights_initialization_07.png"
  [24]: ./images/Weights_initialization_08.png "Weights_initialization_08.png"
  [25]: ./images/Weights_initialization_09.png "Weights_initialization_09.png"
  [26]: ./images/Weights_initialization_10.png "Weights_initialization_10.png"
  [27]: ./images/Weights_initialization_11.png "Weights_initialization_11.png"
  [28]: ./images/Weights_initialization_12.png "Weights_initialization_12.png"
  [29]: ./images/Weights_initialization_01.png "Weights_initialization_01.png"
  [30]: ./images/BN_01.png "BN_01.png"
  [31]: ./images/BN_02.png "BN_02.png"
  [32]: ./images/BN_03.png "BN_03.png"
