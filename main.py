import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def AVG(group):
    mean_val = group.mean()
    group.loc["group"] = mean_val
    re = group.sort_values(by="group", axis=1, ascending=False)
    re = re.iloc[:, :10]
    return re.columns

def AU(group):
    group.loc["group"] = group.sum()
    re = group.sort_values(by="group", axis=1, ascending=False)
    re = re.iloc[:, :10]
    return re.columns

def BC(group):
    group.loc["group"] = (group.rank(axis=1) - 1.0).sum()
    re = group.sort_values(by="group", axis=1, ascending=False)
    re = re.iloc[:, :10]
    return re.columns

def AV(group,stdard=2.5):
    group.loc["group"] = (group > stdard).sum()
    re = group.sort_values(by="group", axis=1, ascending=False)
    re = re.iloc[:, :10]
    return re.columns

def SC(group):
    group.loc["group"] = (group != 0).sum()
    re = group.sort_values(by="group", axis=1, ascending=False)
    re = re.iloc[:, :10]
    return re.columns

def transform_value(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def CR(group):
    re_series = pd.Series(np.zeros(len(group.columns)), index=group.columns)
    for i in range(len(group.columns)):
        col = group.columns[i]
        x = group.subtract(group[col],axis=0)
        x = x.applymap(transform_value)
        sum_row = x.sum(axis=0)

        s_changed = np.where(sum_row > 0, 1, np.where(sum_row < 0, -1, 0))
        s_changed = pd.Series(s_changed, index=sum_row.index)

        re_series += s_changed

    top_10 = re_series.nlargest(10)
    return top_10.index


def make_print(group_num,col):
    print(str(group_num)+"그룹 top10 영화ID:",end='')
    for ID in col:
        print(ID,end=' ')
    print("\n")

'''
데이터 전처리 단계
'''

file_path = 'ratings.dat'

column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df = pd.read_csv(file_path, sep='::', names=column_names, engine='python')
r_df = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)
movie_id = range(1, 3952 + 1)

r_df = r_df.reindex(columns=movie_id, fill_value=0)

kmeans = KMeans(n_clusters=3, random_state=42)
r_df['Cluster'] = kmeans.fit_predict(r_df)

group1 = r_df[r_df['Cluster'] == 0]
group2 = r_df[r_df['Cluster'] == 1]
group3 = r_df[r_df['Cluster'] == 2]
group1 = group1.drop('Cluster', axis=1)
group2 = group2.drop('Cluster', axis=1)
group3 = group3.drop('Cluster', axis=1)


'''
결과값 출력단계
'''

print("### AVG")
avg_group1 = AVG(group1)
avg_group2 = AVG(group2)
avg_group3 = AVG(group3)
make_print(1,avg_group1)
make_print(2,avg_group2)
make_print(3,avg_group3)
print("\n\n")

print("### AU")
au_group1 = AU(group1)
au_group2 = AU(group2)
au_group3 = AU(group3)
make_print(1,au_group1)
make_print(2,au_group2)
make_print(3,au_group3)
print("\n\n")

print("### BC")
bc_group1 = BC(group1)
bc_group2 = BC(group2)
bc_group3 = BC(group3)
make_print(1,bc_group1)
make_print(2,bc_group2)
make_print(3,bc_group3)
print("\n\n")

print("### AV")
av_group1 = AV(group1)
av_group2 = AV(group2)
av_group3 = AV(group3)
make_print(1,av_group1)
make_print(2,av_group2)
make_print(3,av_group3)
print("\n\n")

print("### SC")
sc_group1 = SC(group1)
sc_group2 = SC(group2)
sc_group3 = SC(group3)
make_print(1,sc_group1)
make_print(2,sc_group2)
make_print(3,sc_group3)
print("\n\n")

print("### CR")
cr_group1 = CR(group1)
cr_group2 = CR(group2)
cr_group3 = CR(group3)
make_print(1,cr_group1)
make_print(2,cr_group2)
make_print(3,cr_group3)
