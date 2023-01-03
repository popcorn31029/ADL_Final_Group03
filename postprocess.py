import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os

def main():
    #### delete courses that have been bought

    df_pred = pd.read_csv(args.pred_csv).set_index("user_id")
    dict_pred = df_pred.to_dict("index")

    df_train = pd.read_csv(args.train_csv).set_index("user_id")
    dict_train = df_train.to_dict("index")

    df_val = pd.read_csv(args.val_csv).set_index("user_id")
    dict_val = df_val.to_dict("index")

    courses_final = []
    for user_id in list(dict_pred.keys()):
        # check if the user has taken the course in train.csv
        courses_taken = []
        if user_id in list(dict_train.keys()):
            courses_taken += dict_train[user_id]['course_id'].split(" ")
            
        # check if the user has taken the course in val_ssen.csv
        if user_id in list(dict_val.keys()):
            courses_taken += dict_val[user_id]['course_id'].split(" ")
            
        courses_rec = dict_pred[user_id]['course_id'].split(" ")
        courses_final.append([c for c in courses_rec if c not in courses_taken])
            
    # print("courses_final", len(courses_final))  
    # print("id", len(list(dict_pred.keys())))
            
    df_final = pd.DataFrame({'user_id': list(dict_pred.keys()),
                    'course_id': [" ".join(c) for c in courses_final]})
    df_final.to_csv(os.path.join(args.output_dir, args.postprocess_csv), index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_csv",
        required=True,
        type=Path,
        help="Path to the predict csv file.",
        # default="pred.csv",
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        help="Path to the train csv file.",
        default="./hahow/data/train.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=Path,
        help="Path to model checkpoint.",
        default="./hahow/data/val_seen.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to output directory.",
        default="./pred",
    )
    parser.add_argument(
        "--postprocess_csv",
        type=Path,
        help="Path to the predict csv file.",
        default="pred_clean.csv",
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main()
    
        
        
        

