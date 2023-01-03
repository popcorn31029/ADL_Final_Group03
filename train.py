import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
# from sklearn.metrics import hamming_loss, accuracy_score
from tqdm import tqdm
import jieba
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os

def main():
    # 拿出 training data 中 59714 個 user interest
    aspects_df = pd.read_csv(args.train_interest_csv)
    index_na = list(aspects_df.loc[pd.isna(aspects_df["interests"]), :].index)
    aspects_df = aspects_df.dropna(subset = ['interests'])

    # 用 jieba 對每個 interest 斷詞
    X = [list(jieba.cut(item)) for item in aspects_df["interests"]]
    X = [" ".join(x) for x in X]

    y = np.load('user_course_matrix.npy')
    y = np.asarray([y[i] for i in range(len(y)) if i not in index_na])
    
    # initializing TfidfVectorizer 
    vetorizar = TfidfVectorizer(max_features=1000)
    # fitting the tf-idf on the given data
    vetorizar.fit(X)
    print("vocabulary", list(vetorizar.vocabulary_.keys()))
    
    # splitting the data to training and testing data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # transforming the data
    X_train_tfidf = vetorizar.transform(X_train)
    # X_test_tfidf = vetorizar.transform(X_test)

    # using Multi-label kNN classifier
    mlknn_classifier = MLkNN(k=args.k)
    mlknn_classifier.fit(X_train_tfidf, y_train)

    # 拿出 val/test user 的 interest
    df_val = pd.read_csv(args.test_csv).set_index('user_id')
    dict_val = df_val.to_dict("index")

    df = pd.read_csv(args.course_csv)
    course_id_list = list(df['course_id'].values)

    queries = []
    for id in tqdm(dict_val.keys()):
        # only pick interest, no gender or occupation
        query = dict_val[id]['interests']
        start = query.find('興趣:')
        end = query.find(',喜好:')
        if end == -1:
            query = query[start+3:]
        else:
            query = query[start+3:end]
            
        queries.append(query)
            
    # print(accuracy_score(y_test, predicted))
    # print(hamming_loss(y_test, predicted))    

    new_sentence_tfidf = vetorizar.transform(queries)
    predicted_prob = mlknn_classifier.predict_proba(new_sentence_tfidf).toarray()
    # print("predicted_prob", predicted_prob[0])
    sort_index = [np.argsort(p)[::-1][:args.num_courses] for p in predicted_prob]
    # print("sort_index", sort_index[0])
    courses = [[str(course_id_list[id]) for id in ids] for ids in sort_index]
    # print("courses", courses[0])
        
        
    df_pred = pd.DataFrame({'user_id': list(dict_val.keys()),
                    'course_id': [" ".join(c) for c in courses]})
    df_pred.to_csv(os.path.join(args.output_dir, args.pred_csv), index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_csv",
        type=Path,
        help="Path to the predict csv file.",
        default="pred.csv",
    )
    parser.add_argument(
        "--train_interest_csv",
        type=Path,
        help="Path to the csv file storing train users interest.",
        default="./hahow/data/processed_user_and_group.csv",
    )
    parser.add_argument(
        "--course_csv",
        type=Path,
        help="Path to the preprocessed course csv file.",
        default="./hahow/data/courses.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        help="Path to the preprocessed validation/test csv file.",
        default="./hahow/data/processed_test_unseen_group_2.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to output directory.",
        default="./pred",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="number of the knn cluster",
        default=100,
    )
    parser.add_argument(
        "--num_courses",
        type=int,
        help="number of the courses to recommend",
        default=50,
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main()