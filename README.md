
# AIO Rookie
Here is a step for running this app
## Installation
Should clone repo in 'AIHCM' folder
Copy 'bin' folder to 'AIHCM/dict'
 ```bash
    git clone https://github.com/leducthanhkim2004/aio_update.git
    cd aio_update
    pip install -r requirements.txt
```

If you want run this app on local 
# Please replace 2 line  by your suitable  directory  
```typescript
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
```
Finally run 
```bash
    python home.py 
```

If you want run demo app 

# Please modify 2 lines 
```typescript
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
```
```typescript
link_bin_v2_file = r"/mnt/f/Luan/AIHCM/Workspace/VN_Multi_User_Video_Search/dataset_extraction/faiss_clipv2_cosine.bin" >>>> your "faiss_clipv2_cosine.bin" location, r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
link_bin_file = r"/mnt/f/Luan/AIHCM/Workspace/VN_Multi_User_Video_Search/dataset_extraction/faiss_clip.bin"
json_file = r"/mnt/f/Luan/AIHCM/dict/id2img_fps.json"
```
Finally run on Terminal :
```bash

streamlit run app_demo.py
```
