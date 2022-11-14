import os
import numpy as np
import pdb

from UCP.core import *

def detect_CP_tracks(es_signals):

    print("========Detecting change point from individual ES track===========")

    all_scores_cp_track = []
    all_peaks_cp_track = []

    for each_signal in es_signals:
        res_scores_track, res_peaks_track = detect_cp(each_signal)

        all_scores_cp_track.append(res_scores_track)
        all_peaks_cp_track.append(res_peaks_track)

    return all_peaks_cp_track, all_scores_cp_track


# if __name__ == "__main__":
#     # es_signals should follow the same format with ff
#     # ff = np.load("/home/nttung/research/Monash_CCU/mini_eval/visual_module/test.npy")

#     # load test file
#     es_signals = np.load("../ES_extractor/test_es_feature.npy", allow_pickle=True)
#     all_peaks_track, all_scores_track = detect_CP_tracks(es_signals)
#     pdb.set_trace()

