import faiss
import clip
import open_clip
import numpy as np
import json
import torch

from utils.ocr_retrieval import ocr_retrieval
from utils.combine_utils import merge_searching_results_by_addition
from utils.nlp_processing import Translation

class textSearch:
    def __init__(self, clipv2_bin: str, clip_bin: str, keyframe_file_json: str):
        self.clip_index = self.load_bin_file(clip_bin)
        self.clipv2_index = self.load_bin_file(clipv2_bin)
        self.id2img_fps = self.load_json_file(keyframe_file_json)
        # Load model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/16", device=self._device)
        self.clipv2_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', device=self._device, pretrained='datacomp_xl_s13b_b90k')
        self.clipv2_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.ocr_retrieval = ocr_retrieval()
        self.translater = Translation()

    def load_bin_file(self, bin_file):
        return faiss.read_index(bin_file)

    def load_json_file(self, json_file):
        with open(json_file, 'r') as f:
            js = json.load(f)
        return js

    def find_video_info(self, index_images):
        for keys, values in self.id2img_fps.items():
            if int(keys)== index_images:
                return values
        return None

    def text_search(self, queries: str, k: int, model_type: str, index=None):
        """Implement text search in database"""
        queries = self.translater(queries)
        
        if model_type == 'clip':
            queries = clip.tokenize([queries]).to(self._device)  
            text_features = self.clip_model.encode_text(queries)
            index_choosed = self.clip_index
        else:
            queries = self.clipv2_tokenizer([queries]).to(self._device)  
            text_features = self.clipv2_model.encode_text(queries)
            index_choosed = self.clipv2_index

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)
        
        if index is None:
            scores, index_image = index_choosed.search(text_features, k=k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            scores, index_image = index_choosed.search(text_features, k=k, params=faiss.SearchParametersIVF(id_selector))

        idx_image = index_image.flatten()
        infos_query = [self.find_video_info(idx) for idx in idx_image]  # Ensure idx is a valid index
        link_paths= [info["image_path"] for info in infos_query]
        return scores.flatten(), idx_image, link_paths
    
    def ocr_search(self, ocr_input, k, index=None):
        '''
        Example:
        inputs = {
            'bbox': "a0person",
            'class': "person0, person1",
            'color':None,
            'tag':None
        }
        '''
        scores, idx_image = [], []

        ###### SEARCHING BY OCR #####
        if ocr_input is not None:
            ocr_scores, ocr_idx_image = self.ocr_retrieval(ocr_input, k=k, index=index)
            scores.append(ocr_scores)
            idx_image.append(ocr_idx_image)
        
        scores, idx_image = merge_searching_results_by_addition(scores, idx_image)

        ###### GET INFOS KEYFRAMES_ID ######
        idx_image = idx_image.flatten()
        infos_query = [self.find_video_info(idx) for idx in idx_image]  # Ensure idx is a valid index
        link_paths= [info["image_path"] for info in infos_query]
        return scores.flatten(), idx_image, link_paths
    

# text_searcher = textSearch(link_bin_file, json_file)
# input_test = "A scene from a radiation incident response exercise. The first shot shows a person in yellow and blue lying on the ground wearing a mask, followed by a fire crew using a fire extinguisher to spray smoke. The final shot shows two people in blue protective suits carrying a victim on a stretcher."
# scores, idx_image , image_path = text_searcher.text_search(input_test, k=10, index=None)  # Ensure all three values are returned
# input_test = text_searcher.translater(input_test)
# ocr_input = ""
# lst_scores, list_ids, _, list_image_paths = textSearch.context_search(ocr_input=ocr_input)

# print("Image Indexes:", idx_image)

# print("Image path ", image_path)