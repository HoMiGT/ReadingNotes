# 第1章 欢迎来到Transformer的世界
## 1.1 编码器-解码器框架
`"Transformers are great!" -> 编码器 -> 状态 -> 解码器 -> "Transformer sind grossartig!"`    
简单优雅，弱点：编码器的最终隐藏状态会产生信息瓶颈，必须表示整个输入序列的含义，因为解码器在生成输出时只能靠它来读取全部内容。从而很难处理长序列。    
解决瓶颈的方式：注意力机制，允许解码器访问编码器的所有隐藏状态。
## 1.2 注意力机制
为每个表示都生成一个状态，即解码器可以访问编码器所有隐藏状态。通过设置不同的权重或“注意力”，来按优先级使用状态。    
```
Transformers -> 编码器 -> 状态1                -> Transformers
are          -> 编码器 -> 状态2   =>  解码器    -> sind
great        -> 编码器 -> 状态3                -> grossartig
!            -> 编码器 -> 状态4                -> !
```
仍有重大缺点： 计算本质上是顺序的，不能跨输入序列并行化。 

transformer引入了一种新的建模范式： 自注意力    
自注意力：允许注意力对神经网络同一层中的所有状态进行操作。    
编码器和解码器都有自己的自注意力机制，其输出被馈送到前馈神经网络(FF NN)。    
我们无法访问大量的标注文本数据来训练模型，因此引出Transformer革命的最后一部分： 迁移学习。    
## 1.3 迁移学习
如今，计算机视觉的常见做法是使用迁移学习，即在一项任务上训练像ResNet这样的卷积神经网络，然后在新任务上对其进行适配或微调。    

迁移学习允许网络利用从原始任务中学到的知识。在架构上，这是将模型拆分为主体和头，其中头是针对任务的网络。在训练期间，主体学习来源于领域的广泛特征，并将学习到权重用于初始化新任务的新模型。    

在计算机视觉中，模型首先在包含数百万张图像的[ImageNet](https://image-net.org)等大规模数据集上进行训练。这个训练称为预训练，其主要目的是向模型传授图像的基本特征，例如边或颜色。
然后，这些预训练模型可以在下游任务上进行微调，例如使用相对较少的标注示例(通常每个类几百个)对花种进行分类。微调模型通常比在相同数量的标注数据上从头开始训练的监督模型具有更高的准确率。    

ULMFiT使预训练LSTM模型构建了一个可以适用于各种任务的通用框架：
1. 预训练： 根据前一个单词预测下一个单词，该任务属于语言建模，所生成的模型称为语言模型。
2. 领域适配： 在大规模语料库上进行预训练出语言模型之后，下一步就是将其适配于业务领域语料库。该阶段依然是语言建模方法，但现在模型必须预测目标语料库的下一目标。
3. 微调： 使目标任务的分类层对语言模型进行微调。

GPT仅使用了Transformer架构的解码部分，以及与ULMFiT相同的语言建模方法。    
BERT仅使用了Transformer架构的编码器部分，以及一种称为掩码语言建模的特殊形式的语言建模。    
各个实验室在不同的框架(PyTorch或TensorFlow)上发布模型，导致模型无法移植。于是Hugging Face Transformers库能够跨50多个框架的统一API应用而生。
## 1.4 Hugging Face Transformers库：提供规范化接口
将新颖的机器学习框架应用于新任务通常涉及以下步骤：
1. 将模型架构付诸代码实现(PyTorch或Tensorflow)
2. 从服务器加载预训练权重(如果有)
3. 预处理输入并传给模型，然后应用一些针对具体任务的后处理
4. 实现数据加载器并定义损失函数和优化器来训练模型
## 1.5 Transformer应用概览
1. 文本分类
2. 命名实体识别
3. 问答
4. 文本摘要
5. 翻译
6. 文本生成
## 1.6 Hugging Face生态系统
1. Hugging Face Hub    
迁移学习是推动Transformer成功的关键因素之一。    
Hugging Face Hub 拥有超过20000个免费提供的模型。    
Hub 托管了用于度量指标的数据集和脚本。    
Hub 提供Models和Datasets卡片，用于记录模型和数据集相关内容。

2. Hugging Face Tokenizers    
词元化(tokenization)步骤，该步骤将原始文本拆分为词元的更小部分。
词元可以是单词、单词的一部分或只是标点符号等字符。
Transformer模型是这些词元的数字表示上训练的，因此正确执行**词元化**对于整个NLP项目非常重要！

3. Hugging Face Datasets    
提供智能缓存，通过利用内存映射的特殊机制来突破内存限制，该机制将文件的内容存储到虚拟内存中，并使多个进程能够更有效地修改文件。

4. Hugging Face Accelerate
对训练循环进行细粒度的控制，为常规训练操作增加了一个抽象层以负责训练基础设施所需的所有逻辑。简化基础架构的更改来加速工作流。
## 1.7 Transformer的主要挑战
1. 语言，以英语为主
2. 数据可用性，与人类执行任务所需的量相比，依然差很多
3. 处理长文本，自注意力在段落长度的文本上效果非常好，但在处理整个文档这样的长度的文本，将变得非常昂贵。
4. 不透明度，处理过程不透明，很难或不可能解开模型做出某种预测的“原因”
5. 偏见，主要基于互联网的文本数据进行预训练，将数据中存在的偏见印入模型中。



