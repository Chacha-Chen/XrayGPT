import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION

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
DEBUG = True
## save output to csv
output_path = f'/data/mimic_data/files/mimic-cxr-xraygpt/xraygpt_beam_{0}_temperature_{1}_output_all_validate.csv'.format(BEAM_SIZE, TEMPERATURE)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=False, default='eval_configs/xraygpt_eval.yaml', help="path to configuration file.")
    parser.add_argument("--output-path", required=False, default=output_path, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
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

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img):
    # if gr_img is None:
    #     return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return chat_state, img_list
    # return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    # if len(user_message) == 0:
    #     return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list, llm_message ## why not return conv

title = """<h1 align="center">Demo of XrayGPT</h1>"""
description = """<h3>Upload your X-Ray images and start asking queries!</h3>"""
disclaimer = """ 
            <h1 >Terms of Use:</h1>
            <ul> 
                <li>You acknowledge that the XrayGPT service is designed for research purposes with the ultimate aim of assisting medical professionals in their diagnostic process. It is important to note that the Service does not replace professional medical advice or diagnosis.</li>
                <li>XrayGPT utilizes advanced artificial intelligence algorithms (LLVM's) to carefully analyze and summarize X-ray images for medical diagnostic purposes. The results provided by the Service are derived from the thorough analysis conducted by the AI system, based on the X-ray images provided by the user.</li>
                <li>We strive to provide accurate and helpful results through XrayGPT. However, it is important to understand that we do not make any explicit warranties or representations regarding the effectiveness, reliability, or completeness of the results provided. Our aim is to continually improve and refine the Service to provide the best possible assistance to medical professionals.</li>
            </ul>
            <hr> 
            <h3 align="center">Designed and Developed by IVAL Lab, MBZUAI</h3>

            """

def set_example_xray(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_text_input(example_text: str) -> dict:
    return gr.Textbox.update(value=example_text[0])

# test on test image


def test_single_image(img_path):
    # img_path = os.path.join(os.path.dirname(__file__), "images/example_test_images/img1.png")
    chat_state, img_list = upload_img(img_path)
    chatbot = []
    text_input, chatbot, chat_state = gradio_ask('Take a look at this chest x-ray and describe the findings and impression.', chatbot, chat_state)
    chatbot, chat_state, img_list, llm_message = gradio_answer(chatbot, chat_state, img_list, BEAM_SIZE, TEMPERATURE)
    # print(llm_message)
    return llm_message

# test on test image
split_path = '/data/mimic_data/files/mimic-cxr-resized/2.0.0/mimic-cxr-2.0.0-split.csv.gz'
split = pd.read_csv(split_path, compression='gzip')
test_split = split[split['split'] == 'validate']
dicom_ids = test_split['dicom_id'].tolist()


if not os.path.exists(os.path.dirname(output_path)):
    with open(output_path, 'w') as f:
        ## TODO make sure the final output format contains two columns study_id and report
        f.write('dicom_id,study_id,subject_id,report\n')

## iterrow
with tqdm(total=len(test_split)) as pbar:
    for index, row in test_split.iterrows():
    # for dicom_id in dicom_ids:
        dicom_id = row['dicom_id']
        img_path = '/data/mimic_data/files/mimic-cxr-xraygpt/image/' + dicom_id + '.jpg'
        llm_message = test_single_image(img_path)
        # print(llm_message)

        if not DEBUG:
            with open(output_path, 'a') as f:
                f.write(dicom_id + ',' + str(row['study_id']) + ',' + str(row['subject_id']) + ','
                        + llm_message + '\n')
        pbar.update(1)
        if DEBUG and index == 10:
            break



