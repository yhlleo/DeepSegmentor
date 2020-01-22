## Eval

Evaluation tools for segmentation and edge/centerline detection.

### Evaluation

 - Segmentation Metrics:
   - [x] Global Accuracy (G)
   - [x] Class Average Accuracy (C)
   - [x] Mean IOU (I/U)

 - Sensitivity and Specificity Metrics:
   - [x] Precision (P)
   - [x] Recall (R)
   - [x] F-score (F)


### Crack Detection:

Released results:

 > Note: The PyTorch implementation with the same loss achieves lower performances than the Caffe implementation. So, we suggest to set the loss mode as `focal` in the configuration file [`train_deepcrack.sh`](../scripts/train_deepcrack.sh#L12). 

|Outputs|bT|G|C|I/U|P|R|F|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|DeepCrack-BN|0.31|0.9873|0.9196|0.8643|0.8582|0.8456|0.8518|
|DeepCrack-GF|0.48|0.9888|0.9261|0.8778|0.8795|0.8575|0.8684|
|Side-output 1|0.43|0.9836|0.8930|0.8298|0.8208|0.7939|0.8071|
|Side-output 2|0.42|0.9863|0.9093|0.8543|0.8537|0.8250|0.8391|
|Side-output 3|0.36|0.9854|0.9110|0.8482|0.8334|0.8295|0.8315|
|Side-output 4|0.36|0.9823|0.8989|0.8228|0.7886|0.8077|0.7980|
|Side-output 5|0.38|0.9735|0.8814|0.7663|0.6646|0.7807|0.7180|

For comparisons, you can download our predicted images and evaluation files from [google drive](https://drive.google.com/open?id=1lHm75RoJ5bbk0njKY0Bx-swn9n3fjVIf):

```
deepcrack
  |__ evaluation
  |     |__ ...
  |__ test_latest
        |__images
             |__ ...
```

 - `*_image.png`: input images,
 - `*_label_viz.png`: ground truth,
 - `*_fused.png`: outputs of fused layer,
 - `*_gf.png`: refined predictions by guided filter, see the code [`tools/guided_filter.py`](../tools/guided_filter.py),
 - `*_side1.png`: side output 1,
 - `*_side2.png`: side output 2,
 - `*_side3.png`: side output 3,
 - `*_side4.png`: side output 4,
 - `*_side5.png`: side output 5,



**TODO: CRF refinement module will be released soon...**