# 第一章 PyTorch 简介与安装
略
# 第二章 PyTorch 核心模块
## 2.1 PyTorch 模块结构
路径：Python安装路径/Lib/site-packages/torch
- __pycache__ : 存放python解释器生成的字节码，后缀通常pyc/pyo，空间换时间
- _C : pyi 文件，校验数据类型，底层计算代码采用C++编写，并封装成库，供pytorch的python调用。
- include : c++的头文件
- lib : c++的静态和动态链接库
- autograd : **核心模块与概念**, 实现了梯度的自动求导，极大地简化了深度学习研究者开发的工作量，开发者只需要编写前向传播代码，反向传播部分由autograd自动实现，再也不用手动去推导数学公式，然后编写代码了
- nn : **99%开发者使用频率最高的模块，搭建网络层就在nn.modules里面**
- onnx : pytorch模型转换到onnx模型表示的核心模块
- optim : 优化模块，深度学习的学习过程，就是不断的优化，而优化使用的方法函数，都暗藏在了optim文件夹中。常见的优化方法：Adam, SGD, ASGD. 以及非常重要的学习率调整模块，lr_scheduler.py
- utils : 各类常用工具，其中比较关键的是data文件夹，tensorboard文件夹

路径: Python安装路径/Lib/site-packages/torchvision
- datasets : 官方常用的数据集写的**数据读取函数**
- models : 存放了经典的、可复现的、有训练权重参数可下载的视觉模型，例如分类的alexnet,densenet,efficientnet,mobilenet-v1/2/3,resnet等，分割模型、检测模型、视频任务模型、量化模型。这个库中的模型实现，也是可以借鉴学习的好资料，可以模仿它们的代码结构、函数、类的组织。
- ops : 视觉任务特殊的功能函数，例如检测中用到的roi_align,roi_pool,boxes的生成，以及focal_loss实现，都在这里边有实现。
- transformers : 数据增强库，是pytorch自带的图像预处理、增强、转换工具，可以满足日常的需求。Albumentations。

## 2.2 新冠肺炎x光分类
结果不重要，训练过程重要
```Python
# PyTorch 模块结构
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 思考：如何实现模型训练？ 第1步干嘛，第2步干嘛，...，第n步干嘛
    # step 1 : 数据模块： 构建dataset,dataloader,实现对硬盘中数据的读取及设定预处理方法
    # step 2 : 模型模块： 构建神经网网络，用于后续训练
    # step 3 : 优化模块： 设定损失函数与优化器，用于在训练过程中对网络参数的进行更新
    # step 4 : 迭代模块： 循环迭代进行模型训练，数据一轮又一轮的喂给模型，不断优化模型，直到满足期望

    # step 1 数据模块
    global acc_valid

    class COVID19Dataset(Dataset):
        def __init__(self,root_dir,data_df,transform=None):
            """
            获取数据的路径，预处理的方法
            :param root_dir:
            :param txt_path:
            :param transform:
            """
            self.root_dir = root_dir
            self.data_df = data_df
            self.transform = transform
            self.img_info = []  # [(path,label), ... ,]
            self.label_array = None
            self._get_img_info()

        def __getitem__(self,index):
            """
            输入标量index，从硬盘中读取数据，并预处理 to tensor
            :param index:
            :return:
            """
            path_img,label = self.img_info[index]
            img = Image.open(path_img).convert("L")

            if self.transform is not None:
                img = self.transform(img)

            return img,label

        def __len__(self):
            if len(self.img_info) == 0:
                raise Exception(f"data_dir:{self.root_dir} is empty dir!")
            return len(self.img_info)

        def _get_img_info(self):
            """
            实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在list中
            :return:
            """
            self.data_df = self.data_df[self.data_df["folder"] == "images"]
            self.data_df['findtype'] = (self.data_df['finding'] == "Pneumonia/Viral/COVID-19").astype(int)
            data_df = self.data_df[["filename","findtype"]]
            self.img_info = [(os.path.join(self.root_dir,filename),int(findtype)) for filename,findtype in zip(data_df['filename'],data_df['findtype'])]

    root_dir = r"D:\Downloads\covid-chestxray-dataset-master"
    img_dir = os.path.join(root_dir,"images")
    data_path = os.path.join(root_dir,"metadata.csv")
    data_df = pd.read_csv(data_path)
    train_ratio = 0.8
    val_ratio = 0.1
    train_df,other_df = train_test_split(data_df,test_size = (1-train_ratio),random_state = 11)
    # val_df, test_df = train_test_split(other_df,test_size = (1-val_ratio),random_state = 11)
    transforms_func = transforms.Compose([
        transforms.Resize((8,8)),
        transforms.ToTensor()
    ])
    train_data = COVID19Dataset(img_dir,train_df,transform=transforms_func)
    val_data = COVID19Dataset(img_dir,other_df,transform=transforms_func)
    train_loader = DataLoader(dataset=train_data,batch_size=2)
    val_loader = DataLoader(dataset=val_data,batch_size=2)

    # step 2 模块构建
    class TinnyCNN(nn.Module):
        def __init__(self,cls_num=2):
            super(TinnyCNN,self).__init__()
            self.convolution_layer = nn.Conv2d(1,1,kernel_size=(3,3))
            self.fc = nn.Linear(36,cls_num)

        def forward(self,x):
            x = self.convolution_layer(x)
            x = x.view(x.size(0),-1)
            out = self.fc(x)
            return out
    model = TinnyCNN(2)

    # step 3 优化模块
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,gamma=0.1,step_size=50)

    # step 4 迭代模块
    for epoch in range(100):
        # 训练集训练
        model.train()
        for data,labels in train_loader:
            # forward & backward
            outputs = model(data)
            optimizer.zero_grad()
            # 损失函数计算
            loss = loss_f(outputs,labels)
            loss.backward()
            optimizer.step()
            # 计算分类准确率
            _,predicted = torch.max(outputs.data,1)
            correct_num = (predicted == labels).sum()
            acc = correct_num / labels.shape[0]
            print("Epoch: {} Train Loss: {:.2f} Acc: {:.0%}".format(epoch, loss, acc))

        # 验证集验证
        model.eval()
        for data,labels in val_loader:
            # forward
            outputs = model(data)
            # loss计算
            loss = loss_f(outputs,labels)
            # 计算分类准确率
            _,predicted = torch.max(outputs.data,1)
            correct_num = (predicted == labels).sum()
            acc_valid = correct_num / labels.shape[0]
            print("Epoch: {} Valid Loss: {:.2f} Acc: {:.0%}".format(epoch, loss, acc_valid))

        # 添加停止条件
        if acc_valid == 1:
            break

        # 学习率调整
        scheduler.step()

if __name__ == '__main__':
    main()
```

## 2.3 核心数据结构 —— Tensor

torch.Tensor 类      
torch.tensor 函数    

tensor中有一个很重要的数据，就是模型的参数。更新操作需要记录梯度，梯度的记录功能正式被张量所实现的(求梯度是autograd实现的)

torch.autograd.Variable 在 0.4版本以后与 torch.Tensor合并
- data 保存具体的数据，即被包装的Tensor
- grad 对应data的梯度，形状与data一致
- grad_fn 记录创建该Tensor时用的Function, 该Function在反向传播计算中使用，因此是自动求导的关键
- requires_grad 指示是否计算梯度
- is_leaf 指示节点是否为叶子节点，为叶子节点时，反向传播结束，其梯度仍会保存，非叶子节点的梯度被释放，以节省内存。

torch.Tensor
data : 多维数组，最核心的属性，其他属性都为其服务
dtype : 多维数组的数据类型
shape : 多维数组的形状
device : tensor所在的设备，cpu或cuda
grad, grad_fn, is_leaf, requires_grad ： 与torch.autograd.Variable类似


张量：表示一个数值组成的数组，可能有多个维度。具有一个轴的张量对应数学上的向量(vector)。    
具有俩个轴的张量对应数学上的矩阵(matrix);          
```
x = torch.arange(12)  # 创建行向量 x    
x.shape   # 张量(沿每个轴的长度)的形状    
x.size   # 张量中元素的总数，即形状的所有元素乘积。    
x.reshape(3,4)  # 修改张量的形状而不改变元素的数量和元素值    
torch.zeros((2,3,4))  # 创建形状是(2，3，4)且其他元素都是0的张量    
torch.randn(3,4)  # 创建一个形状为(3,4)的张量，其中每个元素都从均值为0，标准差为1的标准高斯分布(正太分布)中随机采样
``` 

适用于同一形状的任意俩个张量上按元素操作。    
标准算术运算符：+ - * / **     
幂运算：torch.exp(x)    
线性代数运算，包括点积和矩阵乘法    

多张张量的连接在一起
```
torch.cat((X,Y),dim=0)    # 按行，纵向拼接    
torch.cat((X,Y),dim=1)    # 按列，横向拼接

X==Y  # 每个元素判断相等，相等处为真，否则该位置为0    
x.sum()  # 求和
```

广播机制
```
a = torch.arange(3).reshape((3,1))    
b = torch.arange(2).reshape((1,2))    
a+b     # 形状不匹配，广播之后为一个更大的 3x2 的矩阵，矩阵a将复制列，
```

索引和切片    
```
x[-1]  # 最后一个索引
x[1:3]  # 第一个元素和最后一个之前的元素
x[0:2,:]  # 超过2个维度的张量
```

节省内存        
```
z = troch.zeros_like(y)     
z[:] = x+y    # 原地更新 不重新分配内存
```

转换为python的其他对象    
```
np_a = x.numpy()    # tensor -> numpy    
y = torch.tensor(np_a)  # numpy -> tensor        
b = torch.tensor([3.5])
b.item()    # 转换为python标量    
float(b)    
int(b)
```

数据的预处理    
用pandas来预处理数据，处理缺失值
```
inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(input.mean())

inputs = pd.get_dummies(inputs,dummy_na=True)  # 对inputs的类别值或离散值，将NaN视为一个类别
```

pandas亦可转换为标量    
```
x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
```

**线性代数**    
标量: 只有一个元素的张量表示 `x=torch.tensor(3.0)`    
向量: 标量组成的列表。 `x=torch.arange(3) # tensor([0,1,2])`


##2.4 张量的相关函数

高频函数使用    

torch.tensor    
torch.tensor(data,dtype=None,device=None,requires_grad=False,pin_memory=False)    
- data(array_like) tensor的初始数据，可以是list,tuple,numpy array,scalar或其他类型
- dtype(torch.dtype,optional) tensor的数据类型，如torch.uint8,torch.float,torch.long等
- device(torch.device,optional) 决定tensor位于cpu还是gpu。如果为None，会采用默认值，默认值在torch.set_default_tensor_type()中设置，默认为cpu
- requires_grad(bool,optional) 决定是否需要计算梯度
- pin_memory(bool, optional) 是否将tensor存在于锁页内存。这与内存的存储方式有关，默认为False

torch.from_numpy    
通过numpy创建tensor，此时创建的tensor和原array共享同一块内存。
```Python
import torch
import numpy as np
arr = np.array([[1,2,3],[4,5,6]])
t_from_numpy = torch.from_numpy(arr)
print("numpy array: ",arr)
print("tensor: ",t_from_numpy)
print("\n修改arr")
```

torch.zeros    
torch.zeros(*size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False)    

torch.zeros_like    
torch.zeros_like(input,dtype=None,layout=None,device=None,requires_grad=False)        
类似的函数    
torch.ones(*size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False) 给定size创建一个全1的tensor            
torch.ones_like(input,dtype=None,layout=None,device=None,requires_grad=False) 依input的size创建全1的tensor            
torch.full(size,fill_value,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False)  给定的size创建一个值全为fill_value的tensor      
torch.full_like(input,fill_value,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False)        
torch.arange(start=0,end,step=1,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False)  创建等差的1维张量，长度为(end-start)/step，需要注意数值区间为[start,end)        
torch.linspace(start,end,steps=100,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False) 创建均分的1维张量，长度为steps，区间为[start,end]         
torch.logspace(start,end,steps=100,base=10.0,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False) 创建对数均分的1维张量，长度为steps，底为base      
torch.empty(*size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False,pin_memory=False) 依size创建空张量，空是指不会进行初始化操作    
torch.empty_like(input,dtype=None,layout=None,device=None,requires_grad=False)     
torch.empty_strided(size,stride,dtype=None,layout=None,device=None,requires_grad=False,pin_memory=False) 依size创建空张量，空不会进行初始化赋值操作

依概率分布创建    
torch.normal(mean,std,out=None) 为每一个元素以给定的mean和std用高斯分布生成随机数        
torch.rand(*size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False) 在区间[0,1)上，均匀分布        
torch.rand_like(input,dtype=None,layout=None,device=None,requires_grad=False)         
torch.randint(low=0,high,size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False) 在区间[low,hight)上，生成整数的均匀分布        
torch.randint_like(input,low=0,high,dtype=None,layout=torch.strided,device=None,requires_grad=False)     
torch.randn(*size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False) 生成形状为size的标准正态分布张量    
torch.randn_like(input,dtype=None,layout=None,device=None,requires_grad=False)     
torch.randperm(n,out=None,dtype=torch.int64,layout=torch.strided,device=None,requires_grad=False) 生成0到-1的随机排列     
torch.bernoulli(input,*,generator=None,out=None) 以input的值为概率，生成伯努利分布(0-1分布，俩点分布)        

张量的操作

|函数|说明|
|:--:|:--:|
|[cat](https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat)|将多个张量拼接在一起，例如多个特征图的融合可用|
|[concat](https://pytorch.org/docs/stable/generated/torch.concat.html#torch.concat)|同cat，是cat()的别名|
|[conj](https://pytorch.org/docs/stable/generated/torch.conj.html#torch.conj)|返回共轭复数|
|[chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk)|将tensor在某个维度上分成n份|
|[dsplit](https://pytorch.org/docs/stable/generated/torch.dsplit.html#torch.dsplit)|类似numpy.dsplit(),将张量按索引或指定的份数进行切分|
|[column_stack](https://pytorch.org/docs/stable/generated/torch.column_stack.html#torch.column_stack)|水平堆叠张量。即第二个维度上增加，等同于torch.hstack|
|[dstack](https://pytorch.org/docs/stable/generated/torch.dstack.html#torch.dstack)|沿第三个轴进行逐像素拼接|
|[**gather**](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather)|**高级索引方法，目标检测中常用于索引bbox。在指定的轴上，根据给定的index进行索引。**|
|[hsplit](https://pytorch.org/docs/stable/generated/torch.hsplit.html#torch.hsplit)|类似numpy.hsplit(),将张量按列进行切分。若传入整数，则按等分划分。若传入list,则按list中元素进行索引。|
|[hstack](https://pytorch.org/docs/stable/generated/torch.hstack.html#torch.hstack)|水平堆叠张量。即第二个维度上增加，等同于torch.column_stack。|
|[index_select](https://pytorch.org/docs/stable/generated/torch.index_select.html#torch.index_select)|在指定的维度上，按索引进行选择数据，然后拼接成新张量。可知道，新张量的指定维度上长度是index的长度。|
|[masked_select](https://pytorch.org/docs/stable/generated/torch.masked_select.html#torch.masked_select)|根据mask（0/1, False/True 形式的mask）索引数据，返回1-D张量。|
|[movedim](https://pytorch.org/docs/stable/generated/torch.movedim.html#torch.movedim)|移动轴。如0，1轴交换：torch.movedim(t, 1, 0) .|
|[moveaxis](https://pytorch.org/docs/stable/generated/torch.moveaxis.html#torch.moveaxis)|同movedim。Alias for torch.movedim().（这里发现pytorch很多地方会将dim和axis混用，概念都是一样的。）|
|[narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html#torch.narrow)|变窄的张量？从功能看还是索引。在指定轴上，设置起始和长度进行索引。例如：torch.narrow(x, 0, 0, 2)， 从第0个轴上的第0元素开始，索引2个元素。x[0:0+2, ...]|
|[nonzero](https://pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero)|返回非零元素的index。torch.nonzero(torch.tensor([1, 1, 1, 0, 1])) 返回tensor([[ 0], [ 1], [ 2], [ 4]])。建议看example，一看就明白，尤其是对角线矩阵的那个例子，太清晰了。|
|[permute](https://pytorch.org/docs/stable/generated/torch.permute.html#torch.permute)|交换轴。|
|[reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape)|变换形状。|
|[row_stack](https://pytorch.org/docs/stable/generated/torch.row_stack.html#torch.row_stack)|按行堆叠张量。即第一个维度上增加，等同于torch.vstack。Alias of torch.vstack().|
|[scatter](https://pytorch.org/docs/stable/generated/torch.scatter.html#torch.scatter)|scatter_(dim, index, src, reduce=None) → Tensor。将src中数据根据index中的索引按照dim的方向填进input中。这是一个十分难理解的函数，其中index是告诉你哪些位置需要变，src是告诉你要变的值是什么。这个就必须配合例子讲解，请跳转到本节底部进行学习。|
|[scatter_add](https://pytorch.org/docs/stable/generated/torch.scatter_add.html#torch.scatter_add)|同scatter一样，对input进行元素修改，这里是 +=， 而scatter是直接替换。|
|[split](https://pytorch.org/docs/stable/generated/torch.split.html#torch.split)|按给定的大小切分出多个张量。例如：torch.split(a, [1,4])； torch.split(a, 2)|
|[squeeze](https://pytorch.org/docs/stable/generated/torch.squeeze.html#torch.squeeze)|移除张量为1的轴。如t.shape=[1, 3, 224, 224]. t.squeeze().shape -> [3, 224, 224]|
|[stack](https://pytorch.org/docs/stable/generated/torch.stack.html#torch.stack)|在新的轴上拼接张量。与hstack\vstack不同，它是新增一个轴。默认从第0个轴插入新轴。|
|[swapaxes](https://pytorch.org/docs/stable/generated/torch.swapaxes.html#torch.swapaxes)|Alias for torch.transpose().交换轴。|
|[swapdims](https://pytorch.org/docs/stable/generated/torch.swapdims.html#torch.swapdims)|Alias for torch.transpose().交换轴。|
|[t](https://pytorch.org/docs/stable/generated/torch.t.html#torch.t)|转置。|
|[take](https://pytorch.org/docs/stable/generated/torch.take.html#torch.take)|取张量中的某些元素，返回的是1D张量。torch.take(src, torch.tensor([0, 2, 5]))表示取第0,2,5个元素。|
|[take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html#torch.take_along_dim)|取张量中的某些元素，返回的张量与index维度保持一致。可搭配torch.argmax(t)和torch.argsort使用，用于对最大概率所在位置取值，或进行排序，详见官方文档的example。|
|[tensor_split](https://pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split)|切分张量，核心看indices_or_sections变量如何设置。|
|[tile](https://pytorch.org/docs/stable/generated/torch.tile.html#torch.tile)|将张量重复X遍，X遍表示可按多个维度进行重复。例如：torch.tile(y, (2, 2))|
|[transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html#torch.transpose)|交换轴。|
|[unbind](https://pytorch.org/docs/stable/generated/torch.unbind.html#torch.unbind)|移除张量的某个轴，并返回一串张量。如[[1], [2], [3]] --> [1], [2], [3] 。把行这个轴拆了。|
|[unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze)|增加一个轴，常用于匹配数据维度。|
|[vsplit](https://pytorch.org/docs/stable/generated/torch.vsplit.html#torch.vsplit)|垂直切分。|
|[vstack](https://pytorch.org/docs/stable/generated/torch.vstack.html#torch.vstack)|垂直堆叠。|
|[where](https://pytorch.org/docs/stable/generated/torch.where.html#torch.where)|根据一个是非条件，选择x的元素还是y的元素，拼接成新张量。看案例可瞬间明白。|
 
scater_    
scater是将input张量中的部分值进行替换    

张量的随机种子

|随机种子|说明|
|:--:|:--:|
|[seed](https://pytorch.org/docs/stable/generated/torch.seed.html#torch.seed)|获取一个随机的随机种子。Returns a 64 bit number used to seed the RNG.|
|[manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed)|手动设置随机种子，建议设置为42，这是近期一个玄学研究。说42有效的提高模型精度。当然大家可以设置为你喜欢的，只要保持一致即可。|
|[initial_seed](https://pytorch.org/docs/stable/generated/torch.initial_seed.html#torch.initial_seed)|返回初始种子。|
|[get_rng_state](https://pytorch.org/docs/stable/generated/torch.get_rng_state.html#torch.get_rng_state)|获取随机数生成器状态。Returns the random number generator state as a torch.ByteTensor.|
|[set_rng_state](https://pytorch.org/docs/stable/generated/torch.set_rng_state.html#torch.set_rng_state)|设定随机数生成器状态。这两怎么用暂时未知。Sets the random number generator state.|

以上均是设置cpu上的张量随机种子，在cuda上是另外一套随机种子，如torch.cuda.manual_seed_all(seed)， 这些到cuda模块再进行介绍，这里只需要知道cpu和cuda上需要分别设置随机种子。

张量的数学操作
- [Pointwise Ops](https://pytorch.org/docs/stable/torch.html#pointwise-ops) 逐元素的操作
- [Reduction Ops](https://pytorch.org/docs/stable/torch.html#reduction-ops) 减少元素的操作
- [Comparison Ops](https://pytorch.org/docs/stable/torch.html#comparison-ops) 对比操作
- [Spectral Ops](https://pytorch.org/docs/stable/torch.html#spectral-ops) 谱操作，如短时傅里叶变换等各类信号处理的函数
- [Other Operations](https://pytorch.org/docs/stable/torch.html#other-operations) 其他
- [BLAS and LAPACK Operations](https://pytorch.org/docs/stable/torch.html#blas-and-lapack-operations) 基础线性代数
