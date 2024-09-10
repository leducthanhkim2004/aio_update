
# AIO Rookie
Here is a step for running this app
## Installation

 ```bash
    git clone https://github.com/leducthanhkim2004/aio_update.git
    cd aio_update
    pip install -r requirements.txt
```

If you want run this app on local 
# Please modify 2 ink by your link directory  
```typescript
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
```
Finally run 
```bash
    python home.py 
```

If you want run demo app 
#PLEASE modify 2 lines
```typescript
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
```
Finally run on Terminal :
```bash

streamlit run app_demo.py
```