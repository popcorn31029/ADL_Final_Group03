# Multilabel
## Download Model
```
bash download.sh

```
## Data Process
```
python .\data_process.py -u data/users.csv -g data/val_seen_group.csv -s data/subgroups.csv -o test.csv
```
#### -u: Path to users.csv.
#### -g: Path to train or val or test groups.csv.
#### -s: Path to subgroups.csv.
#### -o: Path to output csv file.

## Train & Validation
```
python .\group.py -m hfl/chinese-roberta-wwm-ext-large -o output_test -t processed_train_group_2.csv -v processed_val_group_2.csv
```
#### -m: Path to pretrained model or model identifier from huggingface.co/models.
#### -o: Path to save model.
#### -t: Path to processed train file.
#### -v: Path to processed validate file.
#### -gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass.
#### -epoch: Number of epochs.
#### -batch_size: Batch size.
#### -lr: Initial learning rate (after the potential warmup period) to use.
#### -max_length: Tokenizer's max_length.
## Predict
```
python .\predict_group.py -m output_dir2 -t processed_test_group_2.csv -o output_test.csv
```
#### -m: Path to fine tuned model directory.
#### -t: Path to processed test file.
#### -o: Path to output csv file.
#### -max_length: Tokenizer's max_length.
