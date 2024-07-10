# FaceG2E
## [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.pdf)  | [Project Page](https://faceg2e.github.io/) |  [Arxiv](https://arxiv.org/pdf/2312.00375)

This repository is the official implementation of our work [FaceG2E](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.pdf).

> **[Text-Guided 3D Face Synthesis -- From Generation to Editing](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.pdf)** </br>
> Yunjie Wu, Yapeng Meng, Zhipeng Hu, Lincheng Li, Haoqian Wu, Kun Zhou, Weiwei Xu, Xin Yu</br>
> In CVPR2024</br>
> [FUXI AILab, Netease](https://fuxi.163.com/), Hangzhou, China
![teaser](assets/teaser.png)



## Getting Started
---
### **install the environement**
```
conda env create -f faceg2e.yaml
conda activate faceg2e
bash install_extra_lib.sh
```
This implementation is only tested under the device:
- System: Unbuntu 18.04
- GPU: A30
- Cuda Version: 12.0
- Cuda Driver Version: 525.78.01
---

###  **prepare the data and ckpts**
Download our pretrained texture diffusion [ckpts](https://drive.google.com/file/d/1zmch4TioS4drnvccyVCTMwRid0VUVI9X/view?usp=sharing) and put them in `./ckpts` directory.

Download the HIFI3D++ 3DMM files ([AI-NExT-Albedo-Global.mat](https://drive.google.com/file/d/1vSb2EpduRJuIEUOc_aRsbjASSOUAk7tG/view) and [HIFI3D++.mat](https://drive.google.com/file/d/1MBdk5fsUN1paSOszZYXfwTMehq51Z2kY/view?pli=1)) and put them in `./HIFI3D` directory.

---

### **test generation and editing**

#### facial geometry generation
```
bash demo_geometry_generation.sh
```
#### facial texture generation
```
bash demo_texture_generation.sh
```
#### face editing
```
bash demo_editing.sh
```
The results are saved in `exp/demo`.


## Note
---
1. During editing, you need to input a token indice to indicates the token which determines the consistency-preservation mask. If your editing is a global effect to the face, you can input `0` as indice.

2. The weighting parameters in ``demo_editing.sh`` controls editing effect. You can adjust them by yourself. For example, a higher `edit_prompt_cfg` makes editing more obvious, and a higher `w_reg_diffuse` makes unrelated regions more consistent.

## Results
---
Generation of `Scarlett Johansson`, `Cate Blanchett`, and `Tom Cruise`.
![result_1](assets/results/resized_concat123.gif)


Generation of `Neteyam in Avatar`, `Thanos`, and `Kratos`.
![result_1](assets/results/resized_concat789.gif)


Editing of `Make his eye mask blue`, `Make him chubby`, and `Turn his eyemask golden`.
![result_1](assets/results/resized_edit456.gif)

## Contact
---
If you have any questions, please contact Yunjie Wu (jiejiangwu@outlook.com).

## Citation
---
If you use our work in your research, please cite our publication:
```
@inproceedings{wu2024text,
  title={Text-Guided 3D Face Synthesis-From Generation to Editing},
  author={Wu, Yunjie and Meng, Yapeng and Hu, Zhipeng and Li, Lincheng and Wu, Haoqian and Zhou, Kun and Xu, Weiwei and Yu, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1260--1269},
  year={2024}
}
```

## Acknowledgements
---
There are some functions or scripts in this implementation that are based on external sources. We thank the authors for their excellent works.
Here are some great resources we benefit:

- [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for the rendering framework code.
- [Nvdiffrast](https://github.com/NVlabs/nvdiffrast) for differentiable rendering.
- [REALY](https://realy3dface.com/) for 3D Morphable Model.
- [Stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) for SDS code.
- [BoxDiff](https://github.com/showlab/BoxDiff) for token-image cross-attention compuatation.
