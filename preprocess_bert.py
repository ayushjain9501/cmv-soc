import tarfile
import os.path
import pickle
import json

from bz2 import BZ2File
from urllib import request
from io import BytesIO

import utils
from sentence_transformers import SentenceTransformer



fname = "cmv.tar.bz2"
url = "https://chenhaot.com/data/cmv/" + fname

# download if not exists
if not os.path.isfile(fname):
    f = BytesIO()
    with request.urlopen(url) as resp, open(fname, 'wb') as f_disk:
        data = resp.read()
        f_disk.write(data)  # save to disk too
        f.write(data)
        f.seek(0)
else:
    f = open(fname, 'rb')
tar = tarfile.open(fileobj=f, mode="r")

# Extract the file we are interested in
train_fname = "all/train_period_data.jsonlist.bz2"
train_bzlist = tar.extractfile(train_fname)

# Deserialize the JSON list
original_posts_train = [
    json.loads(line.decode('utf-8'))
    for line in BZ2File(train_bzlist)
]

test_fname = "all/heldout_period_data.jsonlist.bz2"
test_bzlist = tar.extractfile(test_fname)

# Deserialize the JSON list
original_posts_test = [
    json.loads(line.decode('utf-8'))
    for line in BZ2File(test_bzlist)
]

with open("./drive/MyDrive/persuasion/original_posts_train.pkl", 'wb') as f :
    pickle.dump(original_posts_train, f)

with open("./drive/MyDrive/persuasion/original_posts_test.pkl", 'wb') as f :
    pickle.dump(original_posts_test, f)

seed = 0
utils.set_seed(seed)
model = SentenceTransformer('bert-base-nli-mean-tokens')
utils.save_bert_embeddings(model, original_posts_train, "embedding_dict")
utils.save_bert_embeddings(model, original_posts_test, "embdeding_dict_test")




    