'''
    For more efficient inference batching
'''
import os
import os.path as osp
import pdb
import json


path_global_mike_batch = '/home/nttung/research/Monash_CCU/mini_eval/sub_data/converted_video'

# first construct dictionary to know which batch storing which video

dict_video_to_batch_idx = {}

for each_b in os.listdir(path_global_mike_batch):
    path_each_b = osp.join(path_global_mike_batch, each_b)
    
    if 'b' not in each_b or osp.isdir(path_each_b) is False:
        continue

    print('Reading batch:', each_b)
    
    for each_vid_name in os.listdir(path_each_b):
        no_ext_name = each_vid_name.split('.')[0]
        dict_video_to_batch_idx[no_ext_name] = each_b

# read stat.json file for processing
json_stat = '/home/nttung/research/Monash_CCU/mini_eval/sub_data/docs/stat.json'

with open(json_stat, 'r') as fp:
    json_data = json.load(fp)


# start to process batching
path_save_bin_batch_record_file = '/home/nttung/research/Monash_CCU/mini_eval/multimodal_module/index_bin_batch_record'

# my pre-determined size bin batch
bin_batch_size = {'500': 20, '1000': 10, '1500': 10, '2000': 10, '2500': 5, '3000': 2,
                    '3500': 2, '4000': 2, '4500': 1, '6500': 1}


path_global_pred_result = '/home/nttung/research/Monash_CCU/mini_eval/multimodal_module/VIDEO_CCU_output'

for each_bin in json_data:
    bin_batch_json = {}

    if each_bin not in bin_batch_size:
        continue
    size_each_batch = bin_batch_size[each_bin]

    list_all_video_bin = json_data[each_bin]["vid_name_list"]
    total_vid = json_data[each_bin]["total_vid"]

    total_batch = int(total_vid / size_each_batch)
    path_bin_pred_result = osp.join(path_global_pred_result, 'bin_'+str(each_bin))

    for i in range(total_batch+1):
        batch_id = 'batch_'+str(i)

        # get list video name for this batch
        if i+1 > total_batch:
            list_vid_name_cur_batch = list_all_video_bin[i*size_each_batch:]
        else:
            list_vid_name_cur_batch = list_all_video_bin[i*size_each_batch:(i+1)*size_each_batch]
        
        # construct list_input_video_path and list_output_json_path
        list_input_video_path = []
        list_output_json_path = []
        
        for each_vid_name in list_vid_name_cur_batch:

            # define which mike's batch this video belong to
            mike_batch = dict_video_to_batch_idx[each_vid_name]
            input_vid_path = osp.join(path_global_mike_batch, mike_batch, each_vid_name+'.mp4')
            output_json_path = osp.join(path_global_pred_result, 'bin_'+str(each_bin), batch_id, each_vid_name+'.json')

            list_input_video_path.append(input_vid_path)
            list_output_json_path.append(output_json_path)

        bin_batch_json[batch_id] = {}
        bin_batch_json[batch_id]['ls_inp_vid_path'] = list_input_video_path
        bin_batch_json[batch_id]['ls_out_json_path'] = list_output_json_path

    # save json batch_bin record for this bin
    path_json_name_save = osp.join(path_save_bin_batch_record_file, 'bin_'+str(each_bin)+'.json')
    with open(path_json_name_save, 'w') as fp:
        json.dump(bin_batch_json, fp, indent=4)


