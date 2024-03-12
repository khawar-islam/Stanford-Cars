# DiffuseMix : Label-Preserving Data Augmentation with Diffusion Models (CVPR'2024)

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="DiffusMix">
</p>

<p align="center">
    <img src="images/diffuseMix_flower102.png" alt="DiffusMix Treasure">
</p>

---

## 🚀 Getting Started
Setup anaconda environment using `environment.yml` file.

```
conda env create --name DiffuseMix --file=environment.yml
conda remove -n DiffuseMix --all # In case environment installation faileds
```

## 📝 List of Prompts 
Below is the list of prompts, if your accuracy is low then you can use all prompts to increase the performance. Remember that each prompt takes a time to generate images, so the best way is to start from two prompts then increase the number of prompts.

```
prompts = ["Autumn", "snowy", "watercolor art","sunset", "rainbow", "aurora",
               "mosaic", "ukiyo-e", "a sketch with crayon"]
```

## 📁 Dataset Structure
```
train
 └─── class 1
          └───── n04355338_22023.jpg
 └─── class 2
          └───── n03786901_5410.jpg
 └─── ...
```
## ✨ DiffuseMix Augmentation
To introduce the structural complexity, you can download fractal image dataset from here [Fractal Dataset](https://drive.google.com/drive/folders/19xNHNGFv-OChaCazBdMOrwdGRsXy2LPs/)
```
`python3 main.py --train_dir PATH --fractal_dir PATH --prompts sunset,Autumn
```

## 💬 Citation
If you find our work useful in your research please consider citing our paper:
```
@article{diffuseMix2024,
  title={DIFFUSEMIX: Label-Preserving Data Augmentation with Diffusion Models},
  author={Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```