# 引言
据估计，超过80%的数据是非结构化的。以文本、图像、音频、视频等形式存在。文档又占非结构化数据的50%。    
涉及到的Python库：
- NLTK: 自然语言工具包
- SpaCy: 涵盖自然语言处理，并加入深度学习框架
- TextBlob: 基于NLTK和Pattern，受欢迎，但也不是最快或最完整的
- CoreNLP: Stanford CoreNLP的Python的包装器。为各种语言提供文本标记、句法分析和文本分析的技术。

11种自然语言处理的应用
- 情感分析: 客户对企业提供的产品的情感
- 主题建模: 从一组文档中提取唯一的主题
- 投诉分类/邮件分类/电子商务产品分类等
- 使用不同的聚类技术进行文档分类和管理
- 使用相似度方法进行简历筛选和职位描述匹配
- 利用先进的特征工程技术(word2vec和fastText)来捕获上下文
- 信息/文档检索系统，例如搜索引擎
- 聊天机器人、问答(Q&A)以及语音到文本的应用程序(Siri和Alexa)
- 使用神经网络进行语言检测和翻译
- 使用图形方法和先进技术进行文本总结
- 使用深度学习算法进行文本生成/预测下一个单词序列

# 第1章 提取数据
## 方法1-1 适用API收集文本数据
问题：使用推特API收集文本数据    
实现：    
```Python
# pip install tweepy
import numpy as np
import tweepy
import json
import pandas as pd
from tweepy import OAuthHandler
# 验证
consumer_key = "adjbiejfaaoeh"
consumer_secret = "had73haf78af"
access_token = "jnsfby5u4yuawhafjeh"
access_token_secret = "jhdfgay768476r"
# 调用api
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# 提供查询的数据, 搜索ABC时将提供最靠前的10个推文
Tweets = api.search(query, count=10, lang='en', exclude='retweets', tweet_mode='extended')
```
## 方法1-2 从PDF中收集数据
问题：读取一个pdf文件    
实现：
```Python
# pip install PyPDF2
import PyPDF2
from PyPDF2 import PdfFileReader
# 创建一个pdf的read object
pdf = open('file.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf)
print(pdf_reader.numPages)
page = pdf_reader.getPage(0)
# 打印pdf的内容
print(page.extractText())
pdf.close()
```
## 方法1-3 从Word文件中收集数据
问题：读取一个word文件    
实现： 
```Python
# pip install docx
from docx import Document
doc = open('file.docx','rb')
document = Document(doc)
content = ""
for para in document.paragraphs:
    content += para.text
print(content)
```
## 方法1-4 从Json中收集数据
问题：读取一个json文件/对象    
实现： 
```Python
# pip install requests
import request
import josn
r = requests.get("https://quotes.rest/qod.json")
res = r.json()
print(json.dumps(res,indent=4))
# 提取内容
q = res['content']['quotes'][0]
print(q)
```
## 方法1-5 从HTML中收集数据
问题：读取/解析HTML页面    
实现：
```Python
# pip install bs4
# 关于爬虫，scrapy框架以及Playwright、Selenium
import urllib.request as urllib2
from bs4 import BeautifulSoup
response = urllib2.urlopen("https://en.wikipedia.org/wiki/Natural_language_processing")
html_doc = response.read()
# 解析
soup = BeautifulSoup(html_doc, 'html.parser')
# 格式化
strhtm = soup.prettify()
print(strhtm[:1000])
print(soup.title)
print(soup.title.string)
print(soup.a.string)
print(soup.b.string)

for x in soup.find_all('a'):
    print(x.string)

for x in soup.find_all('p'):
    print(x.text)
```
## 方法1-6 使用正则表达式解析文本
问题：希望使用正则表达式来解析文本数据    
补充：
- 基本标志
  - re.I 忽略大小写
  - re.L 查找本地从属项
  - re.M 通过多行找处对象
  - re.S 查找点匹配
  - re.U 采用统一字符编码标准的数据
  - re.X 以更可读的格式编写正则表达式
- 表达式功能
  1. 查找单词出现的字符 [ab]
  2. 查找除a和b以外的字符 [^ab]
  3. 查找a到z之间的字符 [a-z]
  4. 查找a到z以外的字符 [^a-z]
  5. 匹配任意单个字符 .
  6. 匹配任意空白字符 \s
  7. 匹配任意非空白字符 \S
  8. 匹配任意数字 \d
  9. 匹配任意非数字 \D
  10. 匹配任意词 \w
  11. 匹配任意非词 \W
  12. 匹配a或b (a|b)
  13. 匹配0次或1次，且不超过1次的正则 ?
  14. 出现0次及以上 *
  15. 出现1次及以上 +
  16. 出现3次 {3}
  17. 出现3次及以上 {3,}
  18. 出现3到6次 {3,6}
  19. 开始字符串 ^
  20. 结束字符串 $
  21. 匹配词的边界 \b
  22. 匹配非词的边界 \B 
- re.match与re.search的区别
  - re.match 只检查字符串开头是否匹配
  - re.search 检查字符串任意位置是否匹配
实现：    
```Python
# Python内置库 re
import re
# 词汇单元化
re.split('\s+','I like this book.')  # output ['I','like','this','book.']
# 提取邮件IDs
doc = 'For more detials please email us at: xyz@abc.com,pqr@mno.com'
addresses = re.findall(r'[\w\.-]+@[\w\.-]+', doc)  
for address in addresses:
    print(address)  # output xyz@abc.com     pqr@mno.com
# 替换邮件IDs
doc = 'For more details please mail us at xyz@abc.com'
new_doc = re.sub(r'([\w\.-]+)@(\w\.-)+', r'pqr@mno.com', doc)
```
## 方法1-7 处理字符串
问题：希望探索字符串处理    
补充：
- s.find(t) 返回s中字符串t第一次出现的位置的索引，没有找到返回-1
- s.rfind(t) 最后一次出现
- s.index(t) 同 s.find(t) 未找到报 ValueError
- s.rindex(t) 同 s.rfind(t) 未找到报 ValueError
- s.join(text) 拼接字符串
- s.split(t) 拆分字符串，默认以空格
- s.splitlines(t) 将每一行拆分成一系列字符串
- s.lower() 小写
- s.upper() 大写
- s.title() 首字母大写
- s.strip() 剔除字符串开头和结尾的空格
- s.replace(t,u) 将s中的t用u替换

实现：
```Python
# 替换内容
str_v1 = 'I am exploring NLP'
print(str_v1[0])
print(str_v1[5:14])
str_v2 = str_v1.replace('exploring','learning')
print(str_v2)
# 连接俩个字符串
s1 = 'nlp'
s2 = 'machine learning'
s3 = s1+s2
# 搜索字符串中的子串
var = 'I am learning NLP'
f = 'learn'
var.find(f)  # output 5
```
## 方法1-8 从网页抓取文本
问题：从网页中抓取数据，以IMDB网站抓取排名靠前的电影为例    
实现：
```Python
# pip install bs4
# pip install requests
# 代码省略 每个网页都不同
```

# 第2章 探索和处理文本数据
预处理包括将原始文本数据转换成可理解的格式。
## 方法2-1 将文本数据转换为小写形式
问题：将文本数据转换成小写形式    
实现：
```Python
# lower()
import pandas as pd
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good langauage','I like this book','I want more books like this']
df = pd.DataFrame({'tweet':text})
x = 'Testing'
x2 = x.lower()

df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x.lower() for x in s.split()))
```
## 方法2-2 删除标点符号
问题：希望从文本数据中删除标点符号    
解决：
```Python
# 最简单实用正则表达式和replace()函数
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good langauage','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
import re
s = 'I.like.This book!'
s1 =  sub(r'[^\w\s]','',s)

# 或
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')

# 或
import string
s = 'I.like.This book!'
for c in string.punctuation:
    s = s.replace(c,'')
```
## 方法2-3 删除停止词
问题：希望从文本数据中删除停止词    
实现：
```Python
# pip install nltk
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good language','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
import nltk
nltk.download()
for nltk.corpus import stopwords
stop = stopwords.words('english')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))
print(df['tweet'])
```
## 方法2-4 文本标准化
问题：希望实现文本标准化    
实现：
```Python
# 创建一个自定义字典
lookup_dict = {'nlp': 'natural language processing', 'ur': 'your', 'wbu': 'what about you'}
import re
def text_std(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        word = re.sub(r'[^\w\s]','',word)
        if word.lower in lookup_dict:
            word = lookup_dict[word.lower()]
            new_words.append(word)
            new_text = ' '.join(new_words)
    return new_text
text_std('I like nlp it\'s ur choice')
```
## 方法2-5 拼写校正
问题：希望拼写矫正    
实现：    
```Python
# pip install TextBlob
text = ['Introduction to NLP','It is likely to be useful, to people','Machine learning is the new electrcity','R is good langauage','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
from textblob import TextBlob
df['tweet'].apply(lambda x:  str(TextBlob(x).correct()))
print(df)  # 纠正了electricity，language

# 自动纠正库
# pip install autocorrect
from autocorrect import spell
print(spell(u'mussage'))  # message
print(spell(u'sirvice'))  # service
```
## 方法2-6 文本分词
问题：希望实现分词    
实现：
```Python
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good language','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
import nltk
nltk.download('punkt_tab')
from textblob import TextBlob
print(TextBlob(df['tweet'][3]).words)  # WordList(['Would','less','hype','around','ai','action','going','forward'])
mystring = 'My favorite animal is cat'
nltk.word_tokenize(mystring)  # ['My','favorite','animal','is','cat']
```
## 方法2-7 词干提取
问题：词干提取    
实现
```Python
# pip install nltk
# 或 pip install textblob
text = ['I like fishing','I eat fish','There are many fishes in pound']
import pandas as pd
df = pd.DataFrame({'tweet':text})
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['tweet'][:5].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
print(df)   #  都提取出了 fish
```
## 方法2-8 词形还原
问题：希望实现词形还原
实现：
```Python
text = ['I like fishing','I eat fish','There are many fishes in pound']
import pandas as pd
df = pd.DataFrame({'tweet':text})
from textblob import Word
df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['tweet'])  # 可以发现fish和fishes 被还原成fish，leaves和leaf都被词形还原成leaf 
```
## 方法2-9 探索文本数据
问题：希望对文本数据进行探索和理解
实现：
```Python
# 读取文本数据
import nltk
from nltk.corpus import webtext
nltk.download('webtext')
wt_sentences = webtext.sents('firefox.txt')
wt_words = webtext.words('firefox.txt')
# 导入必要的库，用于计算单词出现的频率
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
# 计算所有单词在评论中出现的频率
frequency_dist = nltk.FreqDist(wt_words)
sorted_frequency_dist =  sorted(frequency_dist,key=frequency_dist.__getitem__,reverse=True)
# 考虑长度大于3的单词并绘图
large_words = dict([(k,v) for k, v in frequency_dist.items() if len(k)>3])
frequency_dist = nltk.FreqDist(large_words)
frequency_dist.plot(50,cumulative=False)
# pip install wordcloud
from wordcloud import WordCloud
wcloud = WordCloud().generate_from_frequencies(frequency_dist)
import matplotlib.pyplot as plt
plt.imshow(wcloud,interpolation='bilinear')
plt.axis('off')
plt.show()
```
## 方法2-10 建立一个文本预处理流水线
问题：希望构建一个端到端的文本预处理流水线。无论何时相对任何自然语言处理应用流程进行预处理，都可以直接将数据输入这个流水线处理，并获得干净文本数据作为输出。
实现：
```Python
# 省略，例子是推文，无实际参考意义
# 本质就是用函数封装上述的文本处理
```

# 第3章 文本特征工程
## 方法3-1 使用One-Hot编码将文本转换为特征
问题：希望使用One-Hot编码将文本转换为特征    
缺点：未考虑单词出现的频率，每个单词都当成一个特征
实现：
```Python
text = 'I am learning NLP'
import pandas as pd
pd.get_dummies(text.split())  # 即可实现onehot编码
```
## 方法3-2 使用统计向量器将文本转换为特征
问题：如何通过统计向量器将文本转换为特征
缺点：每个单词都当成一个特征
实现：    
```Python
# pip install sklearn
# sklearn 有一个特征抓取函数，可以从文本中提取特征
from sklearn.feature_extraction.text import CountVectorizer
text =['I love NLP and I will learn NLP in 2month']
vectorizer = CountVectorizer()
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vectorizer.vocabulary_)  # {'love':4, 'nlp':5, 'will':6,'learn':3,'in':2,'2month':0} 
print(vector.toarray())  # [[1,1,1,1,1,2,1]]
```
## 方法3-3 生成N-grams
问题：对给定的句子生成N-grames    
实现：
```Python
# 使用textblob生成N-grames
text = 'I am learning NLP'
from textblob import TextBlob
print(TextBlob(text).ngrams(1))
print(TextBlob(text).ngrams(2))
# 基于Bigram的文档特征
from sklearn.feature_extraction.text import CountVectorizer
text = ['I love NLP and I will learn NLP in 2month']
vectorizer = CountVectorizer(ngram_range=(2,2))
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vectorizer.vocabulary_)
print(vector.toarray())
```
## 方法3-4 生成共生矩阵
问题：理解并生成共生矩阵    
实现：
```Python
import numpy as np
import nltk
from nltk import bigrams
import pandas as pd
import itertools
# 创建共生矩阵
def co_occurence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_to_index = {word:i for i, word in enumerate(vocab)}
    # 创建bigrams
    bi_grams = list(bigrams(corpus))
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    co_occurence_matrix = np.zeros((len(vocab),len(vocab)))
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_to_index[current]
        pos_previous = vocab_to_index[previous]
        co_occurence_matrix[pos_current][pos_previous] = count
    co_occurence_matrix = np.matrix(co_occurence_matrix)
    return co_occurence_matrix,vocab_to_index

# 生成共生矩阵
sentences = [['I','love','nlp'],['I','love','to','learn'],['nlp','is','future'],['nlp','is','cool']]
merged = list(itertools.chain.from_iterable(sentences))
vocab_to_index = {word:i for i, word in enumerate(list(set(merged)))}
matrix = co_occurence_matrix(merged)
CoMatrixFinal = pd.DataFrame(matrix[0],index=vocab_to_index,columns=vocab_to_index)
print(CoMatrixFinal)   
```
## 方法3-5 使用哈希向量器
问题：理解并生成一个哈希向量器
实现：
```Python
# 哈希向量器是一种内存高效，缺点是单向，一旦向量化，则无法检索其特征
# pip install scikit-learn
from sklearn.feature_extraction.text import HashingVectorizer
text =['The quick brown forx jumped over the lazy dog.']
vectorizer = HashingVectorizer(n_features=10)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())
```
## 方法3-6 使用词频-逆文档频率将文本转化为特征
问题：使用词频-逆文档频率将文本转换为特征    
实现：    
```Python
# 词频（TF）: 一个单词在句子中出现的次数与句子长度的比值。
# 逆文档频率 (IDF): 每个单词的逆文档频率是文档总行数与该单词在特定文档中出现的行数之比的对数 IDF = log(N/(n+1)) N为文本的总行数，n为单词出现的行数, n+1是为了平滑整个公式
text = ['The quick brown fox jumped over the lazy dog.','The dog','The fox']
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# 以上方法都是基于频率的嵌入或特征
```
## 方法3-7 实现词嵌入
问题：希望实现词嵌入    
实现：
```Python
# 词嵌入是一种特征学习技术，其中词汇表中的词被映射为捕获上下文层次结构的实数向量。
# 词嵌入式基于预测的方法，使用浅层神经网络来训练模型，得到学习权重，并使用这些权重作为向量表示
# word2vec: 用于训练词嵌入的深度学习谷歌框架。
# word2vec: Skip-Gram(单输入多输出) 和 CBOW (多输入单输出)

# Skip-Gram

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# 数据
sentences = [['I', 'love', 'nlp'],
             ['I', 'will', 'learn', 'nlp', 'in', '2month'],
             ['nlp', 'is', 'future'],
             ['nlp', 'saves', 'time', 'and', 'solves', 'lot', 'of', 'industry', 'problems'],
             ['nlp', 'uses', 'machine', 'learning']]

# 训练 Word2Vec 模型
skipgram = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# 保存模型
skipgram.save('skipgram.bin')

# 加载模型
skipgram = Word2Vec.load('skipgram.bin')

# 提取所有词向量
words = list(skipgram.wv.index_to_key)
vectors = [skipgram.wv[word] for word in words]

# 使用 PCA 降维
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("Word2Vec Skip-Gram 2D PCA Projection")
plt.show()


# CBOW

# pip install gensim
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt 

# 训练模型（使用 CBOW：sg=0）
sentences = [['I','love','nlp'],
             ['I','will','learn','nlp','in','2month'],
             ['nlp','is','future'],
             ['nlp','saves','time','and','solves','lot','of','industry','problems'],
             ['nlp','uses','machine','learning']]
cbow = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0)

print(cbow)
print(cbow.wv['nlp'])  # 访问词向量

# 保存模型
cbow.save('cbow.bin')

# 加载模型
cbow = Word2Vec.load('cbow.bin')

# 获取词和词向量
words = list(cbow.wv.index_to_key)
vectors = [cbow.wv[word] for word in words]

# PCA 降维
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("Word2Vec CBOW 2D PCA Projection")
plt.show()
```
## 方法3-8 实现fastText
问题：如何在python中实现fastText
实现：
```Python
# pip install gensim
# fastText 是word2vec的改进版本
import gensim
from gensim.models import FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt 

# 训练模型（使用 CBOW：sg=0）
sentences = [['I','love','nlp'],
             ['I','will','learn','nlp','in','2month'],
             ['nlp','is','future'],
             ['nlp','saves','time','and','solves','lot','of','industry','problems'],
             ['nlp','uses','machine','learning']]
fast = FastText(sentences, vector_size=20, window=3, min_count=1, workers=5, min_n=1,max_n=2)

print(fast)
print(fast.wv['nlp'])  # 访问词向量

# 保存模型
fast.save('fast.bin')

# 加载模型
fast = FastText.load('fast.bin')

# 获取词和词向量
words = list(fast.wv.index_to_key)
vectors = [fast.wv[word] for word in words]

# PCA 降维
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("Word2Vec CBOW 2D PCA Projection")
plt.show()

```

# 第4章 高级自然语言处理
自然语言处理解决方案需要遵循的流程
- 定义问题：是什么
- 深入理解问题：为什么
- 数据需求头脑风暴：列出所有可能的数据源
- 数据收集：文件，网页等
- 文本预处理
- 文本转换为特征
- 机器学习/深度学习：使系统自动学习数据中的模式，无需规划，大多数自然语言处理都是基于此
- 洞察和部署：对接业务，创造最大的价值和影响
## 方法4-1 提取名词短语
问题：希望提取一个名词短语
实现：
```Python
import nltk
from textblob import TextBlob
blob = TextBlob("John is learning natural language processing")
for np in blob.noun_phrases:
    print(np)
```
## 方法4-2 查找文本之间的相似度
问题：希望查找文本/文档之间的相似度
实现：
```Python
# sklearn的余弦相似度
# 文本相似度
documents = ("I like NLP", 
             "I am exploring NLP",
             'I am a beginner in NLP',
             'I want to learn NLP',
             'I like advanced NLP')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix.shape)

print(cosine_similarity(tfidf_matrix[0:1],tfidf_matrix))

# 语音相似度  使用fuzzy库
```
## 方法4-3 词性标注
问题：标注一个句子的词性
实现： 
```Python
# 词性标注是命名实体解析、情感分析、问题回答和词义消歧的基础
# 有俩种方式可以创建标记器
# 基于规则：手动创建规则，标记属于特定词性的单词
# 基于随机：利用隐藏马尔可夫模型来捕捉单词序列并标注序列的概率

# NLTK拥有很好的词性标注模块
text = "I love NLP and I will learn NLP in 2 month"

import nltk 
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stop_words = set(stopwords.words("english"))

tokens = sent_tokenize(text)

for i in tokens:
    words = nltk.word_tokenize(i)
    words = [w for w in words if not w in stop_words]
    tags = nltk.pos_tag(words)
    print(tags)    
```
词性标注模块的简单形式和解释 
- CC：连接词
- CD：基数词
- DT：限定词
- EX：存在句
- FW：外来词
- IN：介词/从属连词
- JJ：形容词
- JJR：形容词的比较级
- JJS：形容词的最高级
- LS：列表标记
- MD：情感助动词
- NN：名词、单数
- NNS：名词、复数
- NNP：专有名词、单数
- NNPS：专有名词、复数
- PDT：前位限定词
- POS：所有格结束词
- PRP：人称代词
- PRP$：所有格式名词
- RB：副词
- RBR：副词比较级
- RBS：副词最高级
- RP：小品词
- TO：作为介词或不定式格式
- UH：感叹词
- VB：动词、基本形式
- VBD：动词、过去式
- VBG：动词、动名词/现在分词
- VBN：动词、过去分词
- VBP：动词、单数、现在时、非第三人称
- VBZ：动词、第三人称单数、现在时
- WDT：wh-限定词 which
- WP：wh-代词 who、what
- WP$：所有格wh-代词 whose
- WRB：wh-疑问副词 where、when

## 方法4-4 从文本中提取实体
问题：希望从文本中识别和提取实体
实现：
```Python
# pip install spacy nltk
# 预先装模型 
# python -m spacy download en_core_web_sm

sent = 'John is studying at Stanford University in California'

import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
from nltk import ne_chunk
from nltk import word_tokenize

# 提取实体
ne_chunk(nltk.pos_tag(word_tokenize(sent)),binary=False)

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Apple is ready to launch new phone worth $10000 in New york time square')
for ent in doc.ents:
    print(ent.text,ent.start_char, ent.end_char,ent.label_)
# 结果准确，可用于任意自然语言处理应用
```
## 方法4-5 从文本中提取主题
问题：从文本中提取或辨识主题
实现：
```Python
doc1 = 'I am learning NLP, it is very interesting and exciting. it includes machine learning and deep learning'
doc2 = 'My father is a data scientist and he is nlp expert'
doc3 = 'My sister has good exposure into android development'
doc_complete = [doc1,doc2,doc3]
# 清洗和预处理
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words("english"))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
doc_clean = [clean(doc).split() for doc in doc_complete]
print(doc_clean)
# [['learning', 'nlp', 'interesting', 'exciting', 'includes', 'machine', 'learning', 'deep', 'learning'],
# ['father', 'data', 'scientist', 'nlp', 'expert'],
# ['sister', 'good', 'exposure', 'android', 'development']]

# 准备文档术语表
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print(doc_term_matrix)
# [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 3), (5, 1), (6, 1)],
# [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1)],
# [(11, 1), (12, 1), (13, 1), (14, 1), (15, 1)]]

# 创建LDA模型
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix,num_topics=3, id2word=dictionary,passes=50)
print(ldamodel.print_topics())
# [(0, '0.233*"learning" + 0.093*"deep" + 0.093*"interesting" + 0.093*"includes" + 0.093*"exciting" + 0.093*"machine" + 0.093*"nlp" + 0.023*"scientist" + 0.023*"expert" + 0.023*"data"'),
# (1, '0.129*"nlp" + 0.129*"father" + 0.129*"scientist" + 0.129*"data" + 0.129*"expert" + 0.032*"good" + 0.032*"android" + 0.032*"exposure" + 0.032*"development" + 0.032*"sister"'),
# (2, '0.129*"sister" + 0.129*"good" + 0.129*"exposure" + 0.129*"development" + 0.129*"android" + 0.032*"nlp" + 0.032*"father" + 0.032*"data" + 0.032*"expert" + 0.032*"scientist"')]

# 对大量数据执行此操作以提取重要的主题。可以在海量的数据上使用相同的代码片段，以获得重要的结果和信息。
```

## 方法4-6 文本分类
问题：使用机器学习进行垃圾-正常邮件分类
实现：
```Python
# 文本分类的应用：
# 情感分析，文档分类，垃圾-正常邮件分类，简历筛选，文档总结。

# 下载邮件数据集 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset#spam.csv 
# import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
# print("Path to dataset files:", path)

import pandas as pd
Email_Data = pd.read_csv("spam.csv",encoding="latin1")
print(Email_Data.columns)
Email_Data = Email_Data[['v1','v2']]
Email_Data = Email_Data.rename(columns={"v1":"Target","v2":"Email"})
print(Email_Data.head())

# 文本处理和特征工程
import numpy as np
import matplotlib.pyplot as plt
import string 
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os 
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes,metrics,svm

Email_Data['Email'] = Email_Data['Email'].apply(lambda x: ' '.join(x for x in x.split()))
stop = stopwords.words("english")
Email_Data["Email"] = Email_Data['Email'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))

st = PorterStemmer()

Email_Data['Email'] = Email_Data['Email'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
Email_Data['Email'] = Email_Data['Email'].apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
print(Email_Data.head()) 

# 拆分训练集和验证集数据
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Email_Data['Email'],Email_Data['Target'])
# tfidf特征
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=5000)
tfidf_vect.fit(Email_Data['Email'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

print(xtrain_tfidf.data)

# 模型训练，是训练任意给定模型的广义函数
def train_model(classifier,feature_vector_train,label,feature_vector_valid, is_neural_net=False):
    # 训练分类模型
    classifier.fit(feature_vector_train,label)
    # 预测
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions,valid_y)

# 朴素贝叶斯训练
accuracy = train_model(naive_bayes.MultinomialNB(alpha=0.2),xtrain_tfidf,train_y,xvalid_tfidf)
print("Accuracy: ",accuracy)

accuracy = train_model(linear_model.LogisticRegression(),xtrain_tfidf,train_y,xvalid_tfidf)
print("Accuracy: ",accuracy)

# 结果显示朴素贝叶斯的效果更好
```
## 方法4-7 情感分析
问题：希望实现情感分析    
实现：
```Python
# 使用TextBlob情感分析，俩个指标
# 极性：介于[-1,1]的范围内，1表示积极，-1表示消极
# 主观性：公众的主观意识，而非事实信息，取值介于[0,1]之间

# 创建样本数据
review = 'I like this phone. screen quality and camera clarity is really good.'
review2 = 'This tv is not good. Bad quality, no clarity, worst experience.'
# 清洗和预处理
# 获得情感得分
# 仅适用于英文
from textblob import TextBlob
blob = TextBlob(review)
print(blob.sentiment)
blob = TextBlob(review2)
print(blob.sentiment)


# 适用于中文
from snownlp import SnowNLP

text = "我今天很开心"
s = SnowNLP(text)
print(s.sentiments)  # 输出值：0~1，越接近1代表情感越积极

# 更精确的 BERT等深度模型
```
## 方法4-8 消除文本二义性
问题：理解如何消除词义消歧    
实现：
```Python
# pywsd 仅支持英文
```
## 方法4-9 语音转换文本
问题：希望将语音转换为文本    
实现：
```Python
# pip install SpeechRecognition PyAudio
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print('Please say something')
    audio = r.listen(source)
    print('Time over,thanks')
    try:
        print("I think you said: " + r.recognize_google(audio))
    except:
        pass
```
## 方法4-10 文本转换语音
问题：希望文本转换为语音    
实现：
```Python
# pip install gTTS
from gtts import gTTS 
convert = gTTS(text='I like this NLP book', lang='en',slow =False)
convert.save('audio.mp3')
```
## 方法4-11 语言翻译
问题：语言翻译    
实现：
```Python
# pip install goslate
import goslate
# text = 'Bonjour le monde'
text = '你好世界'
gs = goslate.Goslate()
translatedText = gs.translate(text,"en")
print(translatedText)  # Hello world
```

# 第5章 自然语言处理的行业应用
以下内容可借鉴的意义不大，均是传统的算法和算法库。    

**方法5-1 消费者投诉分类**
问题：美国消费者金融保护局每周都会向企业发送数千分消费者对金融产品和服务的投诉，要求企业做出回应，根据投诉进行分类    
实现：
```Python
# https://www.kaggle.com/subhassing/exploring-consumer-complaintdata/data
# 数据不可用了
# 步骤：
#     1. 获取数据
#     2. pandas读取数据
#     3. 理解数据
#     4. 分割数据
#     5. 使用词频-逆文档频率(TF-IDF)进行特征工程
#     6. 模型建立和评估
```
为了提高准确性，可以采用以下方法：
- 使用随机森林、支持向量机、GBM、神经网络、朴素贝叶斯等不同算法重复这个过程
- 采用深度学习技术，如RNN和LSTM等
- 每个算法中，都有许多参数可以调整以获得更好的结果。尝试所有可能的组合，然后给出最好的结果。

**方法5-2 实现情感分类**
**方法5-3 应用文本相似度函数**
**方法5-4 文本数据总结**
**方法5-5 文档聚类**
**方法5-6 搜索引擎中的自然语言处理**

# 第6章 基于深度学习的自然语言处理
神经网络：
- 输入层
- 隐藏层
- 输出层

根据不同的问题或数据，函数可以是不同类型的，其被称为激活函数：
- 线形激活函数
- 非线性激活函数
  - Sigmoid或Logit激活函数
  - Softmax函数 与Sigmoid类似，适合多分类问题
  - Tanh函数 (-1,1),其他的和Sigmoid函数相同
  - 线性整流激活函数：ReLU(将任意小于0的值转换为零)。 $[0,+ \infty]$

 --- 
**卷积神经网络(CNN)**    
单输入层、单输出层、多隐藏层的神经网络

    ｜---------- 特征学习 ---------｜------- 分类 -------｜
输入->卷积ReLU->池化->卷机ReLU->池化->平面层->全连接->Softmax

图像看成是 X * Y * Z 数字数组    
卷积层是卷积神经网络的核心，完成大部分的计算操作。卷积运算符又称滤波器    
卷积操作，就是通过在整个图像上滑动滤波器并计算这俩个矩阵之间的点积形成的矩阵称为“卷积特征”，“激活图”，“特征图”

池化层    
是降低维度而不丢失重要信息。这样做为了减少全连接层的大量输入和处理模型所需的大量计算。有助于减小模型的过度拟合。    

平面层、全连接及Softmax层    
最后一层是一个需要特征向量作为输入的稠密层。将卷积的输出转换为特征向量的过程称为平面化。    
全连接层从平面化层获得一个输入，然后给一个N维向量，其中N是类的数量。作用是通过Softmax函数将N维向量转换成每个类的概率，最终将图像分类为特定的类。    

反向传播：训练神经网络    
将输入的图像送入网络，完成正向传播，即在全连接层的正向传播中进行卷积、ReLU和池化操作，并生成每个类的输出概率。根据前馈规则，权重随机分配，完成训练的第一次迭代，并输出随机概率。    
在第一步结束后，网络使用下面的公式在输出层计算误差：    
$总误差=\sum \sqrt{(目标概率-输出概率)^2}$    
现在，反向传播开始计算相对于网络中所有权重的误差梯度，并使用梯度下降更新所有滤波器的值和权重，这将最终使输出误差最小化。    
滤波器数量、滤波器大小和网络框架等参数将在构建网络时确定。滤波器矩阵和连接权重将在每次运行时更新。    
整个过程就是对整个训练集进行重复训练，直至输出误差最小。

---

**递归神经网络(RNN)**    
在文本中，单词的顺序对创造有意义的句子很重要。    
依赖了时间，误差通过展开的隐藏层从最后一个时间戳反向传播到第一个时间戳。允许计算每个时间戳的误差并更新权重值。在隐藏单元具有递归连接的递归网络中读取整个序列，然后生成所需的输出。    

--- 

**长短时记忆**    
与递归神经网络非常相似，区别是这些单元可以学习很长时间间隔内的东西，且可以像计算机一样存储信息。


## 方法6-1 利用深度学习进行信息检索
## 方法6-2 使用深度学习对文本进行分类
## 方法6-3 对邮件使用长短时记忆预测下一个单词/序列







