[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/siva-sankar-a/eva_final_project) 

# EVA Final Project
EVA4 final project repository

# Problem statement 
The problem statement is to identify camels from a desert environment

## Dataset creation
Dataset creation was divided into 4 main steps

- Collection of foreground images

![Foreground image](img/fg.png)

- Collection of background images 
    - 100 images of desert background were used

- Creating mask images

![Mask image](img/mask.jpg)

## Monocular depth mapping

- Creating depth images


Creating depth maps were tried with 224 X 224 image size and satisfactory results were not obtained

![Depth image](img/depth.png)

Image size was increased to 1024 X 1024 to obtain better depth images

![Foreground image](img/fg1.png)

![Background image](img/depth1.png)

- Steps for disparity map creation :-
    - Images with different backgrounds were equalized for brightness with [Adaptive Histogram Equalization](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)
    - Foregrounds were randomly placed and sized between 320 X 320 and 640 X 640 resolution on the 1024 X 1024 background 

# Dataset URLs

The dataset consists of a total of 40000 fg-bg images, bg images, depth maps and masks.

It can be found in the below urls:

- [Background Images](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/bg_cropped.zip)
- Foregrgound Background Images, Depth Maps and Masks
  - [Part 1](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_0.zip)
  - [Part 2](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_1.zip)
  - [Part 3](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_2.zip)
  - [Part 4](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_3.zip)
  - [Part 5](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_4.zip)
  - [Part 6](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_5.zip)
  - [Part 7](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_6.zip)
  - [Part 8](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_7.zip)
  - [Part 9](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_8.zip)
  - [Part 10](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_9.zip)
- [Dataset information file](https://eva-final-project-dataset.s3-ap-southeast-2.amazonaws.com/dataset_info.csv)

# Dataset metrics

| Image Type | Size |  No of channels |  Mean_R | Mean_G | Mean_B | Mean | Std_R | Std_G | Std_B | Std |
| --- | --- | --- | --- | --- | --- |--- | --- | --- | --- | --- | 
| FG - BG Images  | 1024 X 1024 | 3  | 0.4561 | 0.3984 | 0.3620 | NA | 0.2866 | 0.2640 | 0.2774 | NA |
| BG Images  | 1024 X 1024 | 3  | 0.5869 | 0.5174 | 0.4732 | NA | 0.2560 | 0.2398 | 0.2781 | NA |
| Depth Maps | 512 X 512 |1 | NA | NA | NA | 0.0903 |  NA | NA | NA | 0.2781 |
| Masks | 1024 X 1024 | 1 | NA | NA | NA | 0.6289 | NA | NA | NA | 0.2238 |

# Lessons learnt

- Depth mapping not effective due to excessive shadows and occlusions in background
- Reconsidering background image setup
- Depth mapping prediction gives unsatisfactory results for small image of size 224, 224 