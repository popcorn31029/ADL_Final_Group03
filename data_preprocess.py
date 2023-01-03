import pandas as pd
import re
import csv
from tqdm import tqdm

##### output processed_user_and_group with iser interest and one-hot group label #####

def get_data(path):
    data=[]
    with open(path,'r',encoding='utf-8') as f:
        datas=csv.DictReader(f)
        for d in datas:
            data.append(d)
    return data

users=get_data("./hahow/data/users.csv")
train_groups=get_data('./hahow/data/train_group.csv')
subgroups=get_data("./hahow/data/subgroups.csv")

group_name_and_count=[]
for subgroup in subgroups:
    group_name_and_count.append(tuple([subgroup['subgroup_id'],int(0)]))
group_name_and_count=dict(group_name_and_count)

for train_group in train_groups:
    train_group.update(group_name_and_count)
    for i in list(train_group['subgroup'].split(' ')):
        if i!='':
            train_group[i]=1
    train_group.pop('subgroup')

for train_group in tqdm(train_groups):
    for user in users:
        if train_group['user_id']==user['user_id']:
            train_group['interests']=""
            
            if user['interests']!="":
                train_group['interests']=train_group['interests']+user['interests']+","
                
            if train_group['interests']=="":
                train_group['interests']="無"
            elif train_group['interests'][-1]==",":
                train_group['interests']=train_group['interests'][:-1]

key_name=list(train_groups[0].keys())
key_name.remove('user_id')
key_name.remove('interests')

for train_group in train_groups:
    for key in key_name:
        train_group[subgroups[int(key)-1]['subgroup_name']]=train_group.pop(key)

with open('hahow/data/processed_user_and_group.csv','w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(train_groups[0].keys()))
    writer.writeheader()
    for ele in train_groups:
        writer.writerow(ele)