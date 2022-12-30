import argparse
import numpy as np
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
parser = argparse.ArgumentParser()
parser.add_argument(
        "-m",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
parser.add_argument(
        "-o",
        type=str,
        help="Path to save model.",
        required=True,
    )
parser.add_argument(
        "-t",
        type=str,
        help="Path to processed train file.",
        required=True,
    )
parser.add_argument(
        "-v",
        type=str,
        help="Path to processed validate file.",
        required=True,
    )
parser.add_argument(
        "-gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
parser.add_argument(
        "-epoch",
        type=int,
        default=5,
        help="Number of epochs.",
    )
parser.add_argument(
        "-batch_size",
        type=int,
        default=4,
        help="Batch size.",
    )
parser.add_argument(
        "-lr",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
parser.add_argument(
        "-max_length",
        type=int,
        default=256,
        help="Tokenizer's max_length.",
    )
args = parser.parse_args()

data_files = {}
data_files["train"] = args.t
data_files["validation"] = args.v
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
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=args.max_length)
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
batch_size = args.batch_size
metric_name = "f1"
args = TrainingArguments(
    args.o,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.epoch,
    weight_decay=0.01,
    load_best_model_at_end=True,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
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
trainer.train()
trainer.save_model()