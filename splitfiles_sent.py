#   Sentence-level based file splitter
#   Teslin Roys
import os
import pprint
import nltk

nltk.download('punkt')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def clump(lst, n):
    return list(chunks(lst,n))

def clump_sentences(sentences, fraction_sent_per_clump):
    sent_per_clump = int(fraction_sent_per_clump * len(sentences))
    clumps = clump(sentences, sent_per_clump)
    concat_clumps = []
    #concat clumps
    for c in clumps:
         concat_clumps.append(" ".join(c))
    return concat_clumps

DATA_DIR = r'C:\Users\Administrator\Documents\auth_att\chunkify'
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            print(path)
            bf = open(path,'r',encoding='utf-8', errors="ignore")
            os.makedirs(os.path.join(root,file+"_dir"), exist_ok=True)
            sent_text = nltk.sent_tokenize(bf.read()) 
            clumped_sentences = clump_sentences(sent_text, 0.05)
            part_ct = 0
            for clmp in clumped_sentences:
                f= open(os.path.join(root,file+"_"+str(part_ct)+".txt"), 'w', encoding='utf-8')
                f.write(clmp)
                f.close()
                #also save in sub directory
                f= open(os.path.join(root,file+"_dir",file+"_"+str(part_ct)+".txt"), 'w', encoding='utf-8')
                f.write(clmp)
                f.close()
                part_ct = part_ct+1
