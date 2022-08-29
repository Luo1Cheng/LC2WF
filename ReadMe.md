# LC2WF: Learning to Construct 3D Building Wireframes from 3D Line Clouds

![intro](fig/mtd_overview_v2.jpg)

# Introduction

This repository is for paper "LC2WF:Learning to Construct 3D Building Wireframes from 3D Line Clouds". 

Created by Yicheng Luo*, Jing Ren\*,  Xuefei Zhe, Di Kang, Yajing Xu, Peter Wonka and Linchao Bao.



[[arxiv]](https://arxiv.org/abs/2208.11948)  [[Dataset]](#pretrained-models-and-data) [[Models]](#pretrained-models-and-data)  [[Suppl]]()



# Requirements

* torch   1.8.0

* torchvision 0.9.0

* cuda: 10.0



# Pretrained Models And Data

|                  | url                                                          |
| :--------------: | ------------------------------------------------------------ |
| pretrained-model | [[Google Drive]] / [[BaiduYun]](https://pan.baidu.com/s/1QwSpN5o9wLnhHcrr1H6IZg)(code:engt) |
|     dataset      | [[Google Drive]] / [[BaiduYun]](https://pan.baidu.com/s/1kniIVDjgyLIACVze2g4aow )(code:p9kb) |



# Installation

1. Clone repository

```
git clone https://github.com/Luo1Cheng/LC2WF.git 
```



2. Download data and pretrained model.

   

3. Unzip the files.

```
unzip LC2wf_data.zip
unzip pretrained.zip
```

5. To evaluate the model:

```
python train.py
python trainClassify.py
cd eval_results
python ours_eval.py
```



6. Your can see *.obj results in ./eval_results/finalOutOBJ, you can open them with MeshLab