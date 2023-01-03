## Multilabel k Nearest Neighbors

### Environment
```
pip install -r requirements.txt
```

### Data Preprocessing
The way to generate "processed_test_group_2.csv", "processed_test_unseen_group_2.csv" "processed_val_group_2.csv", "processed_val_unseen_group_2.csv" are the same as branch "Multilabel", please refer to the README.md there.

- generate processed_user_and_group.csv
```
python data_preprocess.py
```
All the files above are generated beforehand, thus you can skip the step above.

- download user_course_matrix.npy
https://drive.google.com/file/d/1oxCfXnGkSZCy4F9BNiROGRsxVHlNTij5/view?usp=sharing
Since the file is too large, please download it from the link and put it at the location that the file configuration instructed.


### File Configuration
```
/hahow
    /data
        /preprocessed_course_all.csv
        /processed_test_group_2.csv
        /processed_test_unseen_group_2.csv
        /processed_user_and_group.csv
        /processed_val_group_2.csv
        /processed_val_unseen_group_2.csv
        /train.csv
        /users.csv
        /val_seen.csv
        /courses.csv
        /user_course_matrix.npy
```


### Training
For seen users, num_courses needs to be more than 50. Since we're going to remove the courses that have been taken. You can also set different k(--k {num}) for the MLKNN classifier.
- seen
```
python train.py --test_csv ./hahow/data/processed_test_group_2.csv --num_courses 100
```
- unseen 
```
python train.py --test_csv ./hahow/data/processed_test_unseen_group_2.csv --num_courses 50
```
- After running the script, there will be an output file at pred/pred.csv. Or you can set the --output_dir and --pred_csv, then the output file will be store in {output_dir/pred_csv}.


### Postprocess
```
python postprocess.py --pred_csv ./pred/pred.csv
```
- After running the script, there will be an output file at pred/pred_clean.csv. Or you can set the --output_dir and --pred_csv, then the output file will be store in {output_dir/pred_csv}.


