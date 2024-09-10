import faiss
import open_clip
import numpy as np
import json
import torch

class textSearch:
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

        if index is None:
            scores, index_image = self.clip_index.search(text_features, k=k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            scores, index_image = self.clip_index.search(text_features, k=k, params=faiss.SearchParametersIVF(id_selector))

        idx_image = index_image.flatten()
        infos_query = [self.find_video_info(idx) for idx in idx_image]  # Ensure idx is a valid index
        link_paths= [info["image_path"] for info in infos_query]
        return scores.flatten(), idx_image,link_paths
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
text_searcher = textSearch(link_bin_file, json_file)
input_test = "the women is wearing a red skirt "
scores, idx_image , image_path= text_searcher.text_search(input_test, k=10, index=None)  # Ensure all three values are returned

print("Image Indexes:", idx_image)

print("Image path ", image_path)