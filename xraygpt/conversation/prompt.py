# Chacha 2023.06.07
# Modify the conversation class to only include a prompt and ask the model to generate response 


from PIL import Image

import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
# from transformers import StoppingCriteria, StoppingCriteriaList

# import dataclasses
# from enum import auto, Enum
# from typing import List, Tuple, Any
# from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import StoppingCriteriaSub, StoppingCriteriaList

class PromptResponse:
    """A class that generate response given a prompt."""
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def get_prompt(self, prompt_txt_file):
        with open(prompt_txt_file, 'r') as f:
            prompt = f.read()
        return prompt
    
    def generate_response(self, prompt_txt_file, input_query, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        prompt = self.get_prompt(prompt_txt_file)
        # prompt_token = self.model.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device).input_ids
        # prompt_emb = self.model.llama_model.model.embed_tokens(prompt_token)
        # prompt_emb = prompt_emb.unsqueeze(0)
        embs = self.get_context_emb(prompt, input_query, img_list) ## TODO: implement this function

        # def ask(self, text, conv):
        #     if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
        #             and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
        #         conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        #     else:
        #         conv.append_message(conv.roles[0], text)

        # def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
        #            repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
            # conv.append_message(conv.roles[1], None)    
            # embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate( 
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        ) 
        # TODO what is the different between llama_model generate vs. llmacausalML
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # output_text = output_text.split('###')[0]  # remove the stop sign '###'
        # output_text = output_text.split('Doctor:')[-1].strip()
        # # conv.messages[-1][1] = output_text
        # output_text = output_text.replace("ChatDoctor", "XrayGPT") ### additionally added
        # output_text = output_text.replace("Chat Doctor", "XrayGPT") ### additionally added
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        return image_emb
        # img_list.append(image_emb)
        # conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        # msg = "Received."
        # # self.conv.append_message(self.conv.roles[1], msg)
        # return msg

    def get_context_emb(self, prompt, input_query, img_list):
    # def get_context_emb(self, conv, img_list):
        # prompt = conv.get_prompt()
        prompt = prompt.replace('<INPUT_QUERY>', input_query)
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]] ## img is always 32, 4096
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs


