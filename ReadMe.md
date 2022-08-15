# Introduction

This repository is for paper "LC2WF:Learning to Construct 3D Building Wireframes from 3D Line Clouds"





# Requirements

* torch   1.8.0

* torchvision 0.9.0

* cuda: 10.0



# Installation

1. Clone repository

```
git clone 
```



2. Dowload linecloud data, wireframe groundtruth and pretrained model.

```
链接：https://pan.baidu.com/s/1P0ZWlm0iIOFr53pYP9L-vg 
提取码：pzwo 
```



3. Unzip the dataset.zip to data/

```
unzip dataset.zip
```



5. To evaluate the model:

```
python train.py
python trainClassify.py
cd eval_results
python ours_eval.py
```



6. Your can see *.obj results in ./eval_results/finalOutOBJ, you can open them with MeshLab