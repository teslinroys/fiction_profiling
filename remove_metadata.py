import re
import os

DATA_DIR = r'C:\Users\Administrator\Documents\auth_att\chunkify'
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            print(path)
            bf = open(path,'r',encoding='utf-8', errors="ignore")
            text = re.sub('<[^<]+>', "", bf.read())
            with open(os.path.join(root,"noxml_"+file), 'w', encoding='utf-8') as f:
                f.write(text)