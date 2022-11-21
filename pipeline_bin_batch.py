import os 
import numpy as np 
import cv2
import pdb
import sys
import bbox_visualizer as bbv
import torch
import json
import os.path as osp

from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from dotmap import DotMap
from datetime import datetime



from ES_extractor.visual_feat_optimize import cal_iou, VisualES

from UCP.inference_ucp import detect_CP_tracks

from CP_aggregator.segment_core import UniformSegmentator
from CP_aggregator.aggregator_core import SimpleAggregator


def run_pipeline_single_video(args, ES_extractor):
    cap = cv2.VideoCapture(args.path_test_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Total video frames:", frame_count)
    
    # reconfig argument
    args.length_video_in_frame = frame_count
    args.fps = fps
    args.frame_count = frame_count
    args.total_vid_frame = frame_count

    args.skip_frame = int(fps/args.min_frame_per_second)

    # update args
    ES_extractor.update_args(args)

    start = datetime.now()

    # ES extractor
    es_signals, all_emotion_category_tracks, all_start_end_offset_track = ES_extractor.extract_sequence_frames(args.path_test_video)

    no_cp_confirm1 = False
    no_cp_confirm2 = False
    la = []
    res_stat = []
    res_cp = []
    res_score = []

    if len(es_signals) == 0:
        no_cp_confirm1 = True

    if no_cp_confirm1 is False:
        # UCP Detector
        # es_signals = np.load("../ES_extractor/test_es_feature.npy", allow_pickle=True)
        all_peaks_track, all_scores_track, all_start_end_offset_track = detect_CP_tracks(es_signals, all_start_end_offset_track)
        
        if len(all_peaks_track) == 0:
            no_cp_confirm2 = True

    if no_cp_confirm1 is False and no_cp_confirm2 is False:


        softmax = torch.nn.Softmax(dim=1)
        # Post-processing step: Refining peak index with start_end_offset_track

        all_refined_peaks_track = []
        all_scores_pick_softmax_track = []
        for each_peak_track, each_start_end_offset_track, each_score_track in zip(all_peaks_track, all_start_end_offset_track, all_scores_track):
            start_idx_track = each_start_end_offset_track[0]
            all_refined_peaks_track.append(each_peak_track + start_idx_track)

            # also pick out score scalar value for specific change point location of that track
            score_pick_track = []
            for each_cp_pos in each_peak_track:
                score_pick_track.append(each_score_track[each_cp_pos])

            softmax_score = softmax(torch.Tensor(np.array([score_pick_track])))
            all_scores_pick_softmax_track.append(softmax_score[0].tolist())
        
        # transform all_peaks_track into index-based change point matrix
        binary_cp_matrix = np.zeros((len(all_refined_peaks_track), args.length_video_in_frame))
        score_cp_matrix = np.zeros((len(all_refined_peaks_track), args.length_video_in_frame))

        for idx_track, each_track in enumerate(all_refined_peaks_track):
            for i, each_cp_index in enumerate(each_track):
                binary_cp_matrix[idx_track][each_cp_index] = 1
                score_cp_matrix[idx_track][each_cp_index] = all_scores_pick_softmax_track[idx_track][i]


        # Video Segmentator
        segmentator = UniformSegmentator()
        res_segment_ids = UniformSegmentator.execute(args.num_intervals, args.length_video_in_frame)

        # Change Point Aggregator
        # extract final change point
        aggregator = SimpleAggregator(args)
        res_cp, res_score, stat_total_cp_interval = aggregator.execute(res_segment_ids, binary_cp_matrix, score_cp_matrix, args.max_cp_found)

        # res_cp = [13, 88]

        # write result stat
        
        la = [(a/fps).astype(int).tolist() for a in all_refined_peaks_track]

        # convert res_segment_ids to second-based
        res_segment_ids_second = [int(res/fps) for res in res_segment_ids]

        # convert stat infor to second-based
        res_stat = []

        for a, b in zip(res_segment_ids, stat_total_cp_interval):
            a_second = int(a/fps)
            res_stat.append((a_second, b))

    time_processing = datetime.now() - start

    result = {"final_cp_result": res_cp, 
                "final_cp_llr": res_score,
                "type": "video", 
                "input_path": args.path_test_video,
                "total_video_frame": frame_count, 
                "num_frame_skip": args.skip_frame,
                'time_processing': int(time_processing.total_seconds()),
                "fps": int(fps), 
                "individual_cp_result": la,
                "stat_segment_seconds_total_cp_accum": res_stat
                }

    # save cp result

    with open(args.path_out_json, 'w') as fp:
        json.dump(result, fp, indent=4)


if __name__ == "__main__":

    ##### Defining arguments #####
    # init argument
    args = DotMap()
    args.device = 'cuda'
    args.threshold_face = 0.6
    args.model_name = 'enet_b0_8_best_afew'
    args.threshold_dying_track_len = 30
    args.threshold_iou_min_track = 0.4
    # skip frame info
    args.min_frame_per_second = 3
    # debug mode
    args.max_idx_frame_debug = None
    args.len_face_tracks = 30
    args.num_intervals = 100
    args.max_cp_found = 3
        
    ##### Initialize Model #####

    face_detector = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=args.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=args.model_name, device=args.device)
    
    ES_extractor = VisualES(args)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)

    ##### Read specific bin batch json file #####
    path_bin_batch_file = '/home/nttung/research/Monash_CCU/mini_eval/multimodal_module/index_bin_batch_record/bin_500.json'

    with open(path_bin_batch_file, 'r') as fp:
        bin_batch_record = json.load(fp)

    # define which batch in bin to perform inference
    batch_run = sys.argv[1]

    # check if batch exist in bin
    if batch_run not in bin_batch_record:
        print('!!! Batch string not exist in bin batch record, check again!!!')
    else:
        list_inp_vid_path = bin_batch_record[batch_run]['ls_inp_vid_path']
        list_out_json_path = bin_batch_record[batch_run]['ls_out_json_path']

        # mkdir path if it does not exist
        tmp = list_out_json_path[0].split('/')
        tmp = tmp[:-1]
        global_path_out_batch_bin_json = '/'.join(tmp)

        if osp.isdir(global_path_out_batch_bin_json) is False:
            os.mkdir(global_path_out_batch_bin_json)

        ##### Perform inference on this batch #####
        for path_test_video, path_out_json in zip(list_inp_vid_path, list_out_json_path):
            
            args.path_test_video = path_test_video
            args.path_out_json = path_out_json
            args.batch_run = batch_run


            run_pipeline_single_video(args, ES_extractor)  
