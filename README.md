
<div align=center>
<a href="https://github.com/ShusenTang/Dive-into-DL-PyTorch" target="-blank">
    <img width="500" src="cover.png">
</a>
</div>

本仓库在[《动手学深度学习》](https:/zh.d2l.ai)的[pytorch实现] (https://github.com/ShusenTang/Dive-into-DL-PyTorch) 的基础上略有修改并将其整理为python脚本，并添加了较多的注释，对初学者更加友好。原书有[中文](https:/zh.d2l.ai)和[英文](https://github.com/ShusenTang/Dive-into-DL-PyTorch)两个版本，版本之间的目录编排和内容有一些不同。原书作者：阿斯顿·张、李沐、扎卡里 C. 立顿、亚历山大 J. 斯莫拉以及其他社区贡献者，GitHub地址：https://github.com/d2l-ai/d2l-zh

**Note**: 所有的代码文件都在pycharm的Community版本上实现通过。

**Advice**: 本人在刚开始用的Anaconda里的spyder代码编辑器，但是因为电脑的一些硬件比较落后，不支持spyder的Kite插件，以及代码补全偶尔会失灵，所以就改用了pycharm，用起来舒心一些。但是pycharm的自定义设置不如spyder，以及没有中文版。



## 简介
本仓库主目录下用于存放代码，子目录small_dataset中存放的是代码中用到的小数据集，子目录Datasets本来是用来存放代码中用到的多个相对较大的数据集的，但是为节省空间以及提高下载速度，在这里只是一个空文件夹，在运行相应的python脚本时，会自动下载数据集到该文件夹中。

原书中有一些章节的代码没有整理，尤其是第九章，只整理了9.1，9.2，9.11章节的python脚本，主要原因如下：
1\. 一些章节主要是讲原理，代码不过寥寥竖行，在Console里敲几行就ok了，没必要整理成一个脚本
2\. pytoch实现的原作者没有给出该章节的pytorch实现，或者需要用pytorch实现mxnet的一些库函数。个人觉得没必要，能够理解原理然后使用科学家们编写好的高质量库实现一些想法或功能就够了，除非你要来改善这个库或者自己实现一个库（第九章计算机视觉）
3\. 本人也尝试着按照原书的mxnet代码进行pytorch改写，还参考了英文版的pytorch实现（比较复杂），但是代码的结果对不上，大概是我太菜了



## 食用方法
建议阅读原书的[pytorch实现版本网页书](https://github.com/ShusenTang/Dive-into-DL-PyTorch)，先理解相关的理论知识，然后利用本仓库整理的脚本和注释，动手运行、修改、调试，加深知识理解。对于pytorch实现版本网页书缺少的章节，则阅读原书网页书对应的章节，先理解原理，再不妨试试基于mxnet/gluon的原书代码，运行、修改、调试，加深知识理解。



## 原书地址
中文版：[动手学深度学习](https://zh.d2l.ai/) | [Github仓库](https://github.com/d2l-ai/d2l-zh)       
English Version: [Dive into Deep Learning](https://d2l.ai/) | [Github Repo](https://github.com/d2l-ai/d2l-en)



## 引用
如果您公开使用了这个仓库的内容请引用原书:
```
@book{zhang2019dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{http://www.d2l.ai}},
    year={2020}
}
```
