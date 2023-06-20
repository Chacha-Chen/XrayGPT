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
DEBUG = False
## save output to csv
dir_prefix =  '/data/mimic_data/files/mimic-cxr-xraygpt'
output_file_name = 'xraygpt_beam_{0}_temperature_{1}_output_all_validate.csv'.format(BEAM_SIZE, TEMPERATURE)
# output_path = os.path.join(dir_prefix, file_name)
# output_path = f'{dir_prefix}xraygpt_beam_{0}_temperature_{1}_output_all_validate.csv'.format(BEAM_SIZE, TEMPERATURE)
# output_path = f'/data/mimic_data/files/mimic-cxr-xraygpt/xraygpt_beam_{0}_temperature_{1}_output_all_validate.csv'.format(BEAM_SIZE, TEMPERATURE)

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
# chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================

def test_single_image(img_path):
    # img_path = os.path.join(os.path.dirname(__file__), "images/example_test_images/img1.png")
    # chat_state = CONV_VISION.copy()
    img_list = []
    img_emb = promptresponse.upload_img(img_path)
    # llm_message = chat.upload_img(img_path, chat_state, img_list)
    # chat_state, img_list = upload_img(img_path)
    img_list.append(img_emb)
    # chatbot = []
    ## change this to prompt txt file
    # user_message ='Take a look at this chest x-ray and describe the findings and impression.'
    # user_prompt = promptresponse.get_prompt('prompt.txt')
    prompt_txt_file = '/data/chacha/XrayGPT/prompts/mimic/test_1.txt'
    # input_query = 'Take a look at this chest x-ray and describe the findings and impression.' ## TODO try others
    input_query = ''
    output_text, output_token = promptresponse.generate_response(prompt_txt_file, input_query, img_list, max_new_tokens=300, num_beams=1)
    # chat.ask(user_message, chat_state)
    # chatbot = chatbot + [[user_message, None]]
    # return '', chatbot, chat_state
    # text_input, chatbot, chat_state = gradio_ask('Take a look at this chest x-ray and describe the findings and impression.', chatbot, chat_state)
    # return chatbot, chat_state, img_list, llm_message ## why not return conv
    # llm_message = chat.answer(conv=chat_state,
    #                           img_list=img_list,
    #                           num_beams=BEAM_SIZE,
    #                           temperature=TEMPERATURE,
    #                           max_new_tokens=300,
    #                           max_length=2000)[0]
    # chatbot[-1][1] = llm_message
    # chatbot, chat_state, img_list, llm_message = gradio_answer(chatbot, chat_state, img_list, BEAM_SIZE, TEMPERATURE)
    # print(llm_message)
    return output_text


print('Start generating')   
print('Output path: {}'.format(output_path))
if not os.path.exists(os.path.dirname(output_path)):
    with open(output_path, 'w') as f:
        ## TODO make sure the final output format contains two columns study_id and report
        f.write('dicom_id; study_id; subject_id; report\n')

## iterrow
with tqdm(total=len(test_split)) as pbar:
    count = 0
    for index, row in test_split.iterrows():
        count += 1
        dicom_id = row['dicom_id']
        img_path = '/data/mimic_data/files/mimic-cxr-xraygpt/image/' + dicom_id + '.jpg'
        output_text = test_single_image(img_path)
        if DEBUG:
            print(output_text)
        # print(llm_message)

        if not DEBUG:
            with open(output_path, 'a') as f:
                f.write(dicom_id + ';' + str(row['study_id']) + ';' + str(row['subject_id']) + ';'
                        + output_text + '\n')
        pbar.update(1)
        if DEBUG and count == 10:
            break
