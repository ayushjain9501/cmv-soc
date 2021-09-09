import torch
import pickle
from tqdm import tqdm
import copy

CONTRACTIONS = {
    "'re" : "are",
    "n't" : "not",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he shall",
    "he'll've": "he shall have",
    "he's": "he has",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "hey will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}
    

def cleanText(selftext) :
    lines = [line for line in selftext.splitlines()
             if not line.lstrip().startswith(">")
             and not line.lstrip().startswith("____")
             and not line.lstrip().startswith("&gt")
             and "edit" not in " ".join(line.lower().split()[:2])
            ]
    selftext = " ".join(lines)
    m = ""
    for word in selftext.split() :
        if word.lower() in CONTRACTIONS :
            m += CONTRACTIONS[word.lower()] + " "
        else :
            m += word.lower() + " "      
    return m.strip()

def set_seed(seed):
    torch.manual_seed(seed)

def save_bert_embeddings(model, data, filename):
    embedding_dict = []
    count = 0
    for cur_post in tqdm(data) :
        post = copy.deepcopy(cur_post)
        OP = post["author"]
        if OP == "[deleted]" :
            embedding_dict.append({"valid" : False})
            continue

        cur_dict = {"valid" : True}

        OP_TEXT = cleanText(post["selftext"])
        OP_ID = post["name"]

        labels = [OP_ID]
        texts = [OP_TEXT]

        for c in post["comments"] :
            if "name" in c and "body" in c :
                label = c["name"]
                text = cleanText(c["body"])
                labels.append(label)
                texts.append(text)
        text_embeddings = model.encode(texts)
        
        s = len(labels)
        for i in range(s) :
            cur_dict[labels[i]] = text_embeddings[i]
        
        embedding_dict.append(cur_dict)

        count += 1
        if count % 100 == 0 :
            with open('./drive/MyDrive/persuasion/bert_embed/{}_{}.pkl'.format(filename, count) , 'wb') as f :
                pickle.dump(embedding_dict, f)
            embedding_dict = []

    with open('./drive/MyDrive/persuasion/bert_embed/{}_{}.pkl'.format(filename, count) , 'wb') as f :
        pickle.dump(embedding_dict, f)