# Fine-Grained Image Classification on Stanford Cars (FGVC-Stanford Cars)

### Setup
Setup anaconda environment using `environment.yml` file.

```
conda env create --name DiffuseMix --file=environment.yml
conda remove -n DiffuseMix --all # In case environment installation faileds
```

### Dataset Structure
```
train
 └─── class 1
          └───── n04355338_22023.jpg
 └─── ...

val
 └─── class 1
          └───── n03786901_5410.jpg
 └─── ...
```
### Train Examples

To introduce the structural complexity, you can download fractal image dataset from here [Fractal Dataset](https://drive.google.com/drive/folders/19xNHNGFv-OChaCazBdMOrwdGRsXy2LPs/)
```
`python3 main.py --train_dir PATH --fractal_dir PATH --prompts sunset,Autumn
```
### Comparison with SOTA Mixup

| Method                   | Stanford Cars |
|--------------------------|---------------|
| Vanilla<sub>(CVPR'16)    | 85.52         |
| RA<sub>(NIPS'20)         | 87.79         |
| AdaAug<sub>(ICLR'22)     | 88.49         |
| PuzzleMix<sub>(ICML'20)  | 89.68         |
| Co-Mixup<sub>(ICLR'21)   | 89.53         |
| Guided-AP<sub>(AAAI'23)  | 90.27         |
| **DiffuseMix**           | **91.26**     |



### Test on Validation Set

```
Accuracy of the network on the 8041 test images: 91.23%
```



## Citation
If you find our work useful in your research please consider citing our paper:
```
@article{diffuseMix2024,
  title={DIFFUSEMIX: Label-Preserving Data Augmentation with Diffusion Models},
  author={Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```