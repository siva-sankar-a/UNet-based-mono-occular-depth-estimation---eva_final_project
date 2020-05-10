[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/siva-sankar-a/eva_final_project) 

# EVA Final Project
EVA4 final project repository

# Problem statement 
The problem statement is to identify camels from a desert environment

## Dataset creation
Dataset creation was divided into 4 main steps

- Collection of foreground images

If you want to embed images, this is how you do it:

![Foregroung image](img/fg.png)

- Collection of background images

- Creating mask images

![Mask image](img/mask.jpg)

- Creating depth images

![Depth image](img/depth.png)

Currently reconsidering the problem statement due to the very poor performance of depth mapping 

# Lessons learnt

- Depth mapping not effective due to excessive shadows and occlusions in background
- Reconsidering background image setup