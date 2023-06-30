# Example: python test_on_mimic_prompt.py --gpu-id 3 --prompt-txt-file test_1.txt --output-file-name debug.csv --debug 

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION
from xraygpt.conversation.prompt import PromptResponse
# imports modules for registration
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *

import pandas as pd
from tqdm import tqdm


BEAM_SIZE = 1
TEMPERATURE = 1
SUBSET_LEN = 500
## save output to csv
dir_prefix =  '/data/mimic_data/files/mimic-cxr-xraygpt'
output_file_name = 'xraygpt_beam_{0}_temperature_{1}_output_all_validate.csv'.format(BEAM_SIZE, TEMPERATURE)

# test on test image
split_path = '/data/mimic_data/files/mimic-cxr-resized/2.0.0/mimic-cxr-2.0.0-split.csv.gz'
split = pd.read_csv(split_path, compression='gzip')
test_split = split[split['split'] == 'validate']
dicom_ids = test_split['dicom_id'].tolist()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=False, default='eval_configs/xraygpt_eval.yaml', help="path to configuration file.")
    parser.add_argument("--output-file-name", required=False, default=output_file_name, help="output file.")
    parser.add_argument("--gpu-id", type=int, default=3, help="specify the gpu to load the model.")
    parser.add_argument("--debug", action="store_true", help="debug mode.")
    parser.add_argument("--subset", action="store_true", help="subset of the dataset to use.")
    parser.add_argument("--prompt-txt-file", required=False, default='test_1.txt', help="prompt txt file. stored inside prompts/mimic")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
output_path = os.path.join(dir_prefix, args.output_file_name)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
promptresponse = PromptResponse(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================

def test_single_image(img_path):

    img_list = []
    img_emb = promptresponse.upload_img(img_path)
    img_list.append(img_emb)

    # prompt_txt_file = '/data/chacha/XrayGPT/prompts/mimic/test_1.txt'
    prompt_txt_file = os.path.join(os.path.dirname(__file__),'prompts/mimic', args.prompt_txt_file)
    
    input_query = '' ## addtional prompt that can be input instance specific
    output_text, output_token = promptresponse.generate_response(prompt_txt_file, input_query, img_list, max_new_tokens=300, num_beams=1)

    return output_text


print('Start generating')   
print('Output path: {}'.format(output_path))

if os.path.exists(output_path) and not args.debug:
    os.remove(output_path)


## iterrow
COLUMNS = ['dicom_id', 'study_id', 'subject_id', 'report']

# total_generation_len = len(test_split)
if args.subset:
    total_generation_len = SUBSET_LEN
else:
    total_generation_len = len(test_split)

# save a dict first then convert to dataframe
output_dict = {'dicom_id': [], 'study_id': [], 'subject_id': [], 'report': []}

with tqdm(total=total_generation_len) as pbar:
    count = 0
    for index, row in test_split.iterrows():
       
        dicom_id = row['dicom_id']
        img_path = '/data/mimic_data/files/mimic-cxr-xraygpt/image/' + dicom_id + '.jpg'
        output_text = test_single_image(img_path)
        if args.debug:
            print(output_text)
        # print(llm_message)

        if not args.debug:
            output_dict['dicom_id'].append(dicom_id)
            output_dict['study_id'].append(row['study_id'])
            output_dict['subject_id'].append(row['subject_id'])
            output_dict['report'].append(output_text)
            # with open(output_path, 'a') as f:
            #     f.write(dicom_id + ';' + str(row['study_id']) + ';' + str(row['subject_id']) + ';'
            #             + output_text + '\n')
        pbar.update(1)
        count += 1
        if args.debug and count == 10:
            break

        if args.subset and count == SUBSET_LEN:
            break

if not args.debug:
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(output_path, index=False, columns=COLUMNS, sep='\t')
