import argparse
import csv
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument(
        "-m",
        type=str,
        help="Path to fine tuned model directory.",
        required=True,
    )
args = parser.parse_args()

data_files = {}
data_files["train"] = 'processed_train_group_2.csv'
data_files["validation"] = 'processed_val_group_2.csv'
extension="csv"
dataset = load_dataset(extension, data_files=data_files)
labels = [label for label in dataset['train'].features.keys() if label not in ['user_id', 'interests']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
tokenizer = AutoTokenizer.from_pretrained(args.m)
def preprocess_data(examples):
    # take a batch of texts
    text = examples["interests"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")
model = AutoModelForSequenceClassification.from_pretrained(args.m, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
batch_size = 4
metric_name = "f1"
args = TrainingArguments(
    args.m,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    metric_for_best_model=metric_name,
)
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
text_lst=[]
with open('processed_test_group_2.csv','r',encoding='utf-8')as f:
    ds=csv.DictReader(f)
    for d in ds:
        text_lst.append(d)

with open('output_test.csv','w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'subgroup'])
    for data in tqdm(text_lst):
        id=data['user_id']
        text=data['interests']
        encoding = tokenizer(text, return_tensors="pt",padding="max_length", truncation=True,max_length=512)
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
