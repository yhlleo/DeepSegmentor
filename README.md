# DeepSegmentor
A Pytorch implementation of DeepCrack and RoadNet projects.

## 1.Datasets

 - [Crack Detection Dataset](https://github.com/yhlleo/DeepCrack)
 - [Multi-task Road Detection Dataset](https://github.com/yhlleo/RoadNet)

## 2.Installation

We provide an user-friendly configuring method via [Conda](https://docs.conda.io/en/latest/) system, and you can create a new Conda environment using the command:

```
conda env create -f environment.yml
```

## 3.Training

Download a dataset and copy it into the folder `datasets`, you can use our provided data loading module or rewrite new ones.

//TODO

## 4.Testing

//TODO

## 5.Evaluation

//TODO

## References

If you take use of our datasets or code, please cite our papers:

```
@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}

@article{liu2018roadnet,
  title={RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes from High-Resolution Remotely Sensed Images},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xia, Menghan and Wang, Xingbo and Liu, Yuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={4},
  pages={2043--2056},
  year={2018},
  doi={10.1109/TGRS.2018.2870871}
}
```

If you have any questions, please contact me without hesitations (yahui.cvrs AT gmail.com).
