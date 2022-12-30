import argparse
import csv
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer
import torch
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument(
        "-m",
        type=str,
        help="Path to fine tuned model directory.",
        required=True,
    )
parser.add_argument(
        "-t",
        type=str,
        help="Path to processed test file.",
        required=True,
    )
parser.add_argument(
        "-o",
        type=str,
        help="Path to output csv file.",
        required=True,
    )
parser.add_argument(
        "-max_length",
        type=int,
        default=512,
        help="Tokenizer's max_length.",
    )
args = parser.parse_args()
data_files = {}
data_files["test"] = args.t
extension="csv"
dataset = load_dataset(extension, data_files=data_files)
labels = [label for label in dataset['test'].features.keys() if label not in ['user_id', 'interests']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(args.m)

model = AutoModelForSequenceClassification.from_pretrained(args.m, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)



trainer = Trainer(
    model,
    tokenizer=tokenizer,
)
text_lst=[]
with open(args.t,'r',encoding='utf-8')as f:
    ds=csv.DictReader(f)
    for d in ds:
        text_lst.append(d)

with open(args.o,'w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'subgroup'])
    for data in tqdm(text_lst):
        id=data['user_id']
        text=data['interests']
        encoding = tokenizer(text, return_tensors="pt",padding="max_length", truncation=True,max_length=args.max_length)
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
        outputs = trainer.model(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        indices = [i for i, p in enumerate(probs)]
        indices.sort(key=lambda i: probs[i], reverse=True)
        pred_label=""
        for idx,i in enumerate(indices):
            if idx>0:
                pred_label+=" "
            pred_label+=str(i+1)
        writer.writerow([id,pred_label])


# turn predicted id's into actual label names
