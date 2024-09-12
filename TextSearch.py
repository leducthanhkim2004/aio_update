import faiss
import open_clip
import numpy as np
import json
import torch

class textSearch:
    """
    This class contain all function for retrievealing image based on queries 
    """
    
    def __init__(self, clipv2_bin: str, keyframe_file_json: str):
        self.clip_index = self.load_bin_file(clipv2_bin)
        self.id2img_fps = self.load_json_file(keyframe_file_json)
        # Load model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clipv2_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', device=self._device, pretrained='datacomp_xl_s13b_b90k')
        self.clipv2_tokenizer = open_clip.get_tokenizer('ViT-L-14')

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

    def text_search(self, queries: str, k: int, index=None):
        """Implement text search in database"""
        text = self.clipv2_tokenizer([queries]).to(self._device)
        text_features = self.clipv2_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        
        scores, index_image = self.clip_index.search(text_features, k=k)
        
        idx_image = index_image.flatten()
        infos_query = [self.find_video_info(idx) for idx in idx_image]  # Ensure idx is a valid index
        link_paths= [info["image_path"] for info in infos_query]
        return scores.flatten(), idx_image,link_paths
