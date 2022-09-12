

# LC2WF: Learning to Construct 3D Building Wireframes from 3D Line Clouds

![intro](fig/mtd_overview_v2.jpg)

# Introduction

This repository is for paper "LC2WF:Learning to Construct 3D Building Wireframes from 3D Line Clouds". 

Created by Yicheng Luo*, Jing Ren\*,  Xuefei Zhe, Di Kang, Yajing Xu, Peter Wonka and Linchao Bao.



[[arxiv]](https://arxiv.org/abs/2208.11948)  [[Dataset]](#pretrained-models-and-data) [[Models]](#pretrained-models-and-data)  [[Suppl]]() [[WebPage]]()



# Requirements

* torch   1.8.0
* torchvision 0.9.0
* cuda: 10.0
* python 3.8.8



# Pretrained Models And Data

|                  | url                                                          |
| :--------------: | ------------------------------------------------------------ |
| pretrained-model | [[Google Drive]] / [[BaiduYun]](https://pan.baidu.com/s/1QwSpN5o9wLnhHcrr1H6IZg)(code:engt) |
|     dataset      | [[Google Drive]] / [[BaiduYun]](https://pan.baidu.com/s/1kniIVDjgyLIACVze2g4aow )(code:p9kb) |



# Evaluation

1. Clone repository

```
git clone https://github.com/Luo1Cheng/LC2WF.git 
```



2. Download line cloud data and pre-trained model.



3. Unzip files

```
unzip LC2wf_data.zip
unzip pretrained.zip
```



4. Your directory will be like

|----LC2WF_data

|    |----house

|    |----LineCloud_0130_P123

|    |----test.txt

|    |----train.txt

|----pretrained

|    |----junction.pth

|    |----edge.pth

|...



5. To evaluate the model:

```shell
python train.py --yamlName evalJunc
python trainClassify.py --yamlName evalWireframe
cd eval_results
python ours_eval.py
```



6. The predicted wireframe obj files are in **./eval_results/finalOutOBJ**. You can open them with MeshLab



# Training

1. Clone repository

```
git clone https://github.com/Luo1Cheng/LC2WF.git 
```



2. Download line cloud data.

   

3. Unzip the files.

```
unzip LC2wf_data.zip
```



4. Train junction prediction model first

```python
python train.py --yamlName train
```



5. Change the **load_model** in **config/genPredJunc.yaml** to your **junction_best.pth** which will be saved in log/***/saved_models folder.



6. Generate predicted Junction of train&test dataset

```python
python train.py --yamlName genPredJunc
```



7. Train connectivity prediction model

```python
python trainClassify.py
```



The best model will be saved in log/***/saved_models folder.



# License



# Acknowledgements