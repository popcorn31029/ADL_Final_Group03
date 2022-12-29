import csv
import json
import jsonlines
from tqdm import tqdm
#5f90538970a38a756834a7eb
def get_data(path):
    data=[]
    with open(path,'r',encoding='utf-8') as f:
        datas=csv.DictReader(f)
        for d in datas:
            data.append(d)
    return data
user_path='data/users.csv'
train_group_path='data/val_unseen_group.csv'
subgroup_path='data/subgroups.csv'
users=get_data(user_path)
train_groups=get_data(train_group_path)
subgroups=get_data(subgroup_path)
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
            if user['gender']!="":
                train_group['interests']="性別:"+user['gender']+","
            if user['occupation_titles']!="":
                train_group['interests']=train_group['interests']+"職業:"+user['occupation_titles']+","
            if user['interests']!="":
                train_group['interests']=train_group['interests']+"興趣:"+user['interests']+","
            if user['recreation_names']!="":
                train_group['interests']=train_group['interests']+"喜好:"+user['recreation_names']
        
            if train_group['interests']=="":
                train_group['interests']="無"
            elif train_group['interests'][-1]==",":
                train_group['interests']=train_group['interests'][:-1]
            continue
key_name=list(train_groups[0].keys())
key_name.remove('user_id')
key_name.remove('interests')

for train_group in train_groups:
    for key in key_name:
        train_group[subgroups[int(key)-1]['subgroup_name']]=train_group.pop(key)
# print(list(train_groups[0].keys()))
# print()
# print(train_groups[0])
# print(train_groups[1])
# print(train_groups[2])
# print(train_groups[3])

with open('processed_val_unseen_group_2.csv','w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(train_groups[0].keys()))
    writer.writeheader()
    for ele in train_groups:
        writer.writerow(ele)
        

    





