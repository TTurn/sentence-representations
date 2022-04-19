import torch
from utils.optimizer import MODEL_CLASS, get_bert_config_tokenizer
config, tokenizer = get_bert_config_tokenizer("roberta")
from models.Transformers import PairSupConBert

path = ""
model = PairSupConBert.from_pretrained(path)

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5)*(normB**0.5))

def run(sentence1, sentence2):
    feature1 = tokenizer.batch_encode_plus(
                [sentence1],
                max_length=32,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
    feature2 = tokenizer.batch_encode_plus(
                [sentence2],
                max_length=32,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
    input_ids1 = feature1['input_ids'].to("cpu")
    input_ids2 = feature2['input_ids'].to("cpu")
    attention_mask1 = feature1['attention_mask'].to("cpu")
    attention_mask2 = feature2['attention_mask'].to("cpu")

    with torch.no_grad():
        embeddings1 = model(input_ids=input_ids1, attention_mask=attention_mask1, task_type="evaluate")
        embeddings2 = model(input_ids=input_ids2, attention_mask=attention_mask2, task_type="evaluate")
    embeddings1 = embeddings1.detach().cpu().numpy()
    embeddings2 = embeddings2.detach().cpu().numpy()
    score = cosine_similarity(embeddings1[0], embeddings2[0])
    return score

with open("test.txt","r")as f, open("result.txt","w")as fw:
    for i, line in enumerate(f):
        if i%1000==0:print(i)
        line_lst = line.strip().split("\t")
        if len(line_lst) != 3: continue
        score = run(line_lst[0], line_lst[1])
        fw.write(line_lst[0]+"\t"+line_lst[1]+"\t"+str(score)+"\t"+line_lst[2]+"\n")

