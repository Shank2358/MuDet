# MuDet: Multimodal Collaboration Networks for Geospatial Vehicle Detection in Dense, Occluded, and Large-Scale Events

  <a href="https://github.com/Shank2358/GGHL/">
    <img alt="Version" src="https://img.shields.io/badge/Version-1.3.0-blue" />
  </a>
  
  <a href="https://github.com/Shank2358/GGHL/blob/main/LICENSE">
    <img alt="GPLv3.0 License" src="https://img.shields.io/badge/License-GPLv3.0-blue" />
  </a>
  
<a href="mailto:zhanchao.h@outlook.com" target="_blank">
   <img alt="E-mail" src="https://img.shields.io/badge/To-Email-blue" />
</a> 

## This is the implementation of MuDet 👋👋👋
[[IEEE TGRS]([https://ieeexplore.ieee.org/document/9709203](https://ieeexplore.ieee.org/document/10475352))]

#### 上述代码基于我的另外一个仓库GGHL进行的改进，详细使用方法和说明可以参考那边仓库的readme

  ### Give a ⭐️ if this project helped you. If you use it, please consider citing:
  ```IEEE TGRS
  @ARTICLE{10475352,
  author={Wu, Xin and Huang, Zhanchao and Wang, Li and Chanussot, Jocelyn and Tian, Jiaojiao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multimodal Collaboration Networks for Geospatial Vehicle Detection in Dense, Occluded, and Large-Scale Events}, 
  year={2024},
  volume={62},
  number={},
  pages={1-12},
  keywords={Vehicle detection;Feature extraction;Object detection;Disasters;Streaming media;Remote sensing;Convolutional neural networks;Convolutional neural networks (CNNs);dense and occluded;hard-easy balanced attention;large-scale disaster events;multimodal vehicle detection (MVD);remote Sensing (RS)},
  doi={10.1109/TGRS.2024.3379355}}
  ```

## 🌈 1.Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10)   
CUDA > 11.1, Cudnn > 8.0.4

First, install CUDA, Cudnn, and Pytorch.
Second, install the dependent libraries in [requirements.txt](https://github.com/Shank2358/GGHL/blob/main/requirements.txt). 

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt  
```
    
## 🌟 2.Installation
1. git clone this repository    

2. Polygen NMS  
The poly_nms in this version is implemented using shapely and numpy libraries to ensure that it can work in different systems and environments without other dependencies. But doing so will slow down the detection speed in dense object scenes. If you want faster speed, you can compile and use the poly_iou library (C++ implementation version) in datasets_tools/DOTA_devkit. The compilation method is described in detail in [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) .

```bash
cd datasets_tools/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace 
```   
  
## 🎃 3.Datasets

#### (1) Training Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./dataR folder.  
For the specific format of the train.txt file, see the example in the /dataR folder.   

```txt
image_path xmin,ymin,xmax,ymax,class_id,x1,y1,x2,y2,x3,y3,x4,y4,area_ratio,angle[0,180) xmin,ymin,xmax,ymax,class_id,x1,y1,x2,y2,x3,y3,x4,y4,area_ratio,angle[0,180)...
```  
The calculation method of angle is explained in [Issues #1](https://github.com/Shank2358/GGHL/issues/1) and our paper.

#### (2) Validation & Testing Format
The same as the Pascal VOC Format

#### (3) DataSets Files Structure
  ```
  cfg.DATA_PATH = "/opt/datasets/MuDet/"
  ├── ...
  ├── JPEGImages
  |   ├── 000001.png
  |   ├── 000002.png
  |   └── ...
  ├── Annotations (MuDet Dataset Format)
  |   ├── 000001.txt (class_idx x1 y1 x2 y2 x3 y3 x4 y4)
  |   ├── 000002.txt
  |   └── ...
  ├── ImageSets
      ├── test.txt (testing filename)
          ├── 000001
          ├── 000002
          └── ...
  ```  

## 🌠🌠🌠 4.Usage Example
#### (1) Training  
```python
python train_MuDet.py
```


#### (2) Testing  
```python
python eval_MuDet.py
```
  
## 📝 License  
Copyright © 2021 [Shank2358](https://github.com/Shank2358).<br />
This project is [GNU General Public License v3.0](https://github.com/Shank2358/GGHL/blob/main/LICENSE) licensed.

## 🤐 To be continued 
