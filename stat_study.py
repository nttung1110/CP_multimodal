# manipulate stat from segment id change point produced by multistage model to draw some insights
import os
import numpy as np
import json
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
import matplotlib.patches as mpatches

from visualize import process_and_cp_file


def draw_stat_insight(stat_record, prediction, label, individual_cp, path_save):

    x = []
    y = []
    for a in stat_record:
        x.append(a[0])
        y.append(a[1])

    x = np.array(x)
    y = np.array(y[:-1])
    fig, ax = plt.subplots(2, sharex=True)

    ## draw plot 1
    plt1 = ax[0].bar(x=x[:-1], height=y, width=np.diff(x), align='edge', fc='skyblue', ec='black')
    

    # draw groundtruth cp
    gt_cp = label['cp']

    for each_cp in gt_cp:
        cp_pos = int(each_cp)
        ax[0].axvline(x=cp_pos, color='r')


    # draw prediction cp
    for each_pred in prediction:
        cp_pos = int(each_pred)
        ax[0].axvline(x=cp_pos, color='b')


    ax[0].set_xlabel('Time interval in seconds')
    ax[0].set_ylabel('Change point accumulated count')

    red_patch = mpatches.Patch(color='red', label='Change point from annotators')
    blue_patch = mpatches.Patch(color='blue', label='My prediction change point')

    ax[0].legend(handles=[red_patch, blue_patch])

    ## draw plot 2
    # generating color for different track
    prime_color_list = ['red', 'blue', 'green', 'pink', 'yellow', 'black', 'grey', 'orange']   

    total_individual_cp = len(individual_cp)
  

    if total_individual_cp <= len(prime_color_list):
        color_list = prime_color_list[:total_individual_cp]
    else:
        color_list = prime_color_list*int(total_individual_cp/len(prime_color_list)) # create multiple copies of prime_color_list
        
        if total_individual_cp % len(prime_color_list) != 0:
            remaining = total_individual_cp % len(prime_color_list)
            color_list.extend(prime_color_list[:remaining])



    for idx_track, track_cp in enumerate(individual_cp):
        for each_cp in track_cp:
            ax[1].scatter(each_cp, idx_track, s=1, c=color_list[idx_track])

    ax[1].set_ylabel('Track ID')
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(path_save) 



if __name__ == "__main__":
    path_res = '/home/nttung/research/Monash_CCU/mini_eval/multimodal_module/utils/vis_compare_cp_pred_gt'

    # read and construct groundtruth label
    path_gt = '/home/nttung/research/Monash_CCU/mini_eval/visual_data/cp_annotation_setAB/LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V1.0/data/changepoint.tab'
    dict_gt_label = process_and_cp_file(path_gt)


    path_stat_draw = '/home/nttung/research/Monash_CCU/mini_eval/multimodal_module/output_cp_3/'

    for each_file_name in os.listdir(path_stat_draw):
        print(each_file_name)
        full_path_read = osp.join(path_stat_draw, each_file_name)
        fname_no_tail = each_file_name.split('.')[0]

        # prediction data
        with open(full_path_read, "r") as fp:
            data = json.load(fp)

        path_save = osp.join(path_res, fname_no_tail+'.png')
        stat_record = data['stat_segment_seconds_total_cp_accum']
        prediction = data['final_cp_result']
        label = dict_gt_label[fname_no_tail]
        individual_cp = data['individual_cp_result']


        draw_stat_insight(stat_record, prediction, label, individual_cp, path_save)
