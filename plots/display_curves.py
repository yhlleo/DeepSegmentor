# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Usage:
  python3 display_curves.py --data_dir <path_to_dir> --suffix prf
"""

import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.switch_backend('agg')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./demo', help='path to files') 
    parser.add_argument('--suffix', type=str, default='prf')
    parser.add_argument('--xlabel', type=str, default='Recall')
    parser.add_argument('--ylabel', type=str, default='Precision')
    parser.add_argument('--output', type=str, default='PR-curve.pdf')
    parser.add_argument('--legend_loc', type=str, default='lower left')
    opts = parser.parse_args()

    files = glob.glob(os.path.join(opts.data_dir, "*.{}".format(opts.suffix)))
    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

    for ff in files:
        fname = ff.split('/')[-1].split('.')[0]
        p_acc, r_acc, f_acc = [], [], []
        with open(ff, 'r') as fin:
            for ll in fin:
                bt, p, r, f = ll.strip().split('\t')
                p_acc.append(float(p))
                r_acc.append(float(r))
                f_acc.append(float(f))
        max_index = np.argmax(np.array(f_acc))
        axs.plot(np.array(r_acc), np.array(p_acc), label='[F={:.03f}]{}'.format(f_acc[max_index], fname).replace('=0.', '=.'), lw=2)
    
    axs.grid(True, linestyle='-.')
    axs.set_xlim([0., 1.])
    axs.set_ylim([0., 1.])
    axs.set_xlabel('{}'.format(opts.xlabel))
    axs.set_ylabel('{}'.format(opts.ylabel))
    axs.legend(loc='{}'.format(opts.legend_loc))

    pdf = PdfPages(r'{}'.format(opts.output))
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    pdf.close()
    pdf=None


