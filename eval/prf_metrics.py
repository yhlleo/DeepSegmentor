# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Calculate sensitivity and specificity metrics:
 - Precision
 - Recall
 - F-score
"""

import numpy as np
from data_io import imread

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        print(thresh)
        statistics = []
        
        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt/255).astype('uint8')
            pred_img = (pred/255 > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
        
        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
        # calculate recall
        r_acc = tp/(tp+fn)
        # calculate f-score
        final_accuracy_all.append([thresh, p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc)])
    return final_accuracy_all

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn] 
