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
from moviepy.editor import *




from ES_extractor.visual_feat_collab import cal_iou, VisualES

from UCP.inference_ucp import detect_CP_tracks

from CP_aggregator.segment_core import UniformSegmentator
from CP_aggregator.aggregator_core import SimpleAggregator


def run_pipeline_single_video(args, ES_extractor):
    # cap = cv2.VideoCapture(args.path_test_video)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Total video frames:", frame_count)

    large_clip = VideoFileClip(args.path_test_video)
    large_frame_count = int(large_clip.fps * large_clip.duration)
    large_fps = int(large_clip.fps)
    print("Total clip video frames:", large_frame_count)

    # Fragmenting video for faster processing but there is a trade-off
    length_in_seconds = int(large_frame_count/large_fps)
    n_segment = int(length_in_seconds/args.min_seconds_per_segment)

    final_res_cp, final_res_score, final_la, final_res_stat, final_start_end_segment = [], [], [], [], []
    start = datetime.now()
    print("Total sub clip:", n_segment+1)

    # now iterate and process each segment
    for i in range(n_segment+1):
        start_second = i*args.min_seconds_per_segment

        if start_second >= large_clip.duration:
            break
        
        if i+1 > n_segment:
            end_second = length_in_seconds
        else:
            end_second = (i+1)*args.min_seconds_per_segment

        sub_clip = large_clip.subclip(start_second, end_second)
        sub_frame_count = int(sub_clip.fps * sub_clip.duration)
        sub_fps = int(sub_clip.fps)

        print(('Sub-clip %dth')%int(i+1))
        print('Start second of subclip:', start_second)
        print('End second of subclip:', end_second)
        print("Total sub-clip video frames:", sub_frame_count)
        print("===================")

        # reconfig argument
        args.length_video_in_frame = sub_frame_count
        args.fps = sub_fps
        args.frame_count = sub_frame_count
        args.total_vid_frame = sub_frame_count

        args.skip_frame = int(args.fps/args.min_frame_per_second)

        # update args
        ES_extractor.update_args(args)


        # ES extractor
        es_signals, all_emotion_category_tracks, all_start_end_offset_track = ES_extractor.extract_sequence_frames(sub_clip)

        no_cp_confirm1 = False
        no_cp_confirm2 = False
        la = []
        res_stat = []
        res_cp = []
        res_score = []
        start_end_segment = (start_second, end_second)

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
            
            la = [(a/args.fps).astype(int).tolist() for a in all_refined_peaks_track]

            # convert res_segment_ids to second-based
            res_segment_ids_second = [int(res/args.fps) for res in res_segment_ids]

            # convert stat infor to second-based
            res_stat = []

            for a, b in zip(res_segment_ids, stat_total_cp_interval):
                a_second = int(a/args.fps)
                res_stat.append((a_second, b))

            # STEP FOR SPECIAL TREATMENT, SHIFT CP RESULT TO MATCH INDEX OF EACH SEGMENT => SHIFT res_cp only

            for idx in range(len(res_cp)):
                res_cp[idx] += start_second

        final_res_cp.append(res_cp)
        final_res_score.append(res_score)
        final_la.append(la)
        final_res_stat.append(res_stat)
        final_start_end_segment.append(start_end_segment)

    time_processing = datetime.now() - start
    
    result = {"final_cp_result": final_res_cp, 
                "final_cp_llr": final_res_score,
                "type": "video", 
                "input_path": args.path_test_video,
                "total_video_frame": large_frame_count, 
                "num_frame_skip": args.skip_frame,
                'time_processing': int(time_processing.total_seconds()),
                "fps": int(large_fps), 
                "individual_cp_result": final_la,
                "stat_segment_seconds_total_cp_accum": final_res_stat
                }

    if args.min_seconds_per_segment is not None:
        result['is_special_treatment'] = 1
        result['start_end_seconds_each_segment'] = final_start_end_segment
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
    # THIS PARAMETERS IS ONLY USED FOR SPECIAL VIDEO TREATMENT
    args.min_seconds_per_segment = 480

    # machine inference: jvn or collab
    # args.machine_run = 'jvn'
        
    ##### Initialize Model #####

    face_detector = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=args.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=args.model_name, device=args.device)
    
    ES_extractor = VisualES(args)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)

    # define which batch and which bin to perform inference
    batch_run = sys.argv[1]
    bin_run = sys.argv[2]
    machine_run = sys.argv[3]

    args.machine_run = machine_run

    ##### Read specific bin batch json file #####
    if args.machine_run == 'jvn':
        path_bin_batch_file = './index_bin_batch_record/'+bin_run+'.json'
    else:
        path_bin_batch_file = './index_bin_batch_record_collab/'+bin_run+'.json'

    with open(path_bin_batch_file, 'r') as fp:
        bin_batch_record = json.load(fp)

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
            args.bin_run = bin_run

            print('Processing:', args.path_test_video, args.path_out_json, args.batch_run, args.bin_run)

            if os.path.isfile(args.path_out_json) is True:
                print('...Result file exists, skipping...')
                continue

            run_pipeline_single_video(args, ES_extractor)  
