# SAM-CTMapper-coastal-wetland-hyperspectral-image-classification
Demo code of "SAM-CTMapper: Utilizing segment anything model and scale-aware mixed CNN-Transformer facilitates coastal wetland hyperspectral image classification"


## Step 1: prepare dataset
**GF-5 Yancheng & ZY1-02D Huanghekou**: https://zenodo.org/records/8105220

**Prisma Shuangtai & Prisma Chongming**: [百度网盘，提取码 quja](https://pan.baidu.com/s/1VPapO_ZT_QfrfTcWT6mv7Q?pwd=quja) or [Google Drive](https://drive.google.com/drive/folders/1macw4UJ2ADywohav9AqSxEG-MwiwupNW?usp=sharing)

## Step 2: compiling cuda files
```
cd lib
. install.sh ## please wait for about 5 minutes
```
you can also refer to [ESCNet](https://github.com/Bobholamovic/ESCNet) for the compiling process.

## Step 3: train and test
```
python main_Mix.py
```

## Step 4: record classification result

The quantitative results and qualitative results will be recorded in the '/results' folder.

## Citation
If you find this work interesting in your research, please kindly cite:
```
@article{ZOU2025104469,
title = {SAM-CTMapper: Utilizing segment anything model and scale-aware mixed CNN-Transformer facilitates coastal wetland hyperspectral image classification},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {139},
pages = {104469},
year = {2025},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2025.104469},
}
```
Thank you very much! (*^▽^*)

If you have any questions, please feel free to contact me (Jiaqi Zou, immortal@whu.edu.cn).
