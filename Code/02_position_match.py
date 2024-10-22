# %%
import pandas as pd
import os
import pyarrow.parquet as pq
import glob
import numpy as np
import datetime
import requests
import googlemaps
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

'''
Created by Yanzun Yang
Last edit: 2024/10/21
This file is to match with hospital positions and identify AI positions
'''

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option("expand_frame_repr", False)


# %%
# find the absolute path of the current script
script_dir = os.path.abspath(__file__)
# construct the working directory path
work_dir = os.path.abspath(os.path.join(script_dir, '../../')) # go back to the parent branch
os.chdir(work_dir)
print(work_dir)


# %% Get the ids for hospitals
hospital_list = pd.read_excel('Output_data/hospital_list_company_ref.xlsx')
hospital_list = hospital_list[['rcid']][hospital_list['rcid'].notna()]
hospital_list = hospital_list.drop_duplicates()

# %%
# import individual position data
# define the folder path containing the files
folder_path = 'F:/academic_individual_position/*.parquet'
# get a list of all parquet files in the folder
file_list = glob.glob(folder_path)
# initialize an empty list to hold each DataFrame
df_list = []
# loop through each file and append it to the list
i = 0
for file in file_list:
    print(file)
    parquet = pq.read_table(file)
    df = parquet.to_pandas()
    df = pd.merge(df, hospital_list, on='rcid', how='inner')
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
hospital_position = pd.concat(df_list, ignore_index=True)
del parquet, df_list
'''
parquet = pq.read_table('F:/academic_individual_position/individual_position_0068_part_00.parquet')
df = parquet.to_pandas()
del parquet
hospital_position = pd.read_csv('F:\output_data\hospital_position_original.csv')
'''

# %%
print(hospital_position['country'].value_counts())
a = (hospital_position['company_raw']==hospital_position['company_name']).astype(int)
df['a']=(df['company_raw']==df['company_name']).astype(int) # company_name is similar to company_raw
hospital_position = hospital_position.drop(columns=['company_raw','company_linkedin_url','company_name','ultimate_parent_rcid','ultimate_parent_company_name','naics_code','naics_description'])
hospital_position.to_csv('F:/output_data/hospital_position_original.csv') # 4684 hospitals have positions
print(hospital_position['state'][hospital_position['country']=='empty'].value_counts()) # as long as country is empty, state is also empty
hospital_position = hospital_position[hospital_position['country']=='United States'] # obs num from 2001146 to 1880761; 48978 are empty

# %%
hospital_list = pd.read_excel('Output_data/hospital_list_company_ref.xlsx')
hospital_list = hospital_list[['rcid','Hospital Name','Sate','City']][hospital_list['rcid'].notna()]
US_states = pd.read_excel('Data/US_states.xlsx')
hospital_list = pd.merge(US_states, hospital_list, on='Sate', how='right')
hospital_position = pd.merge(hospital_position, hospital_list, on='rcid', how='inner') # 4656 hospitals have position data
a = hospital_position[['location_raw','state', 'metro_area','Hospital Name','Sate','City','State_name','rcid']]
a.head(1000000).to_excel('Output_data/hospital_position_location1.xlsx', index=False)
a.tail(923998).to_excel('Output_data/hospital_position_location2.xlsx', index=False)

# %% Match companies based on locations
hospital_position['location_raw_'] = hospital_position['location_raw'].str.lower()
hospital_position['city_'] = hospital_position['City'].str.lower()
hospital_position['city_match'] = hospital_position.apply(lambda row: row['city_'] in row['location_raw_'], axis=1)
print(hospital_position['city_match'].value_counts())
'''
True     1039966
False     875866
'''
hospital_position['location1'] = hospital_position['location_raw'].apply(lambda x: x.split(',')[0] if ',' in x else x)
hospital_position['location1'] = hospital_position['location1'].str.lower()
hospital_position['state_name_'] = hospital_position['State_name'].str.lower()
hospital_position['state_match'] = hospital_position.apply(lambda row: row['state_name_'] in row['location1'], axis=1)
print(hospital_position['state_match'].value_counts())
'''
False    1821896
True       93936
'''
hospital_position['metro_area_'] = hospital_position['metro_area'].apply(lambda x: x.split(" metropolitan area")[0] if "metropolitan area" in x else x)
hospital_position['metro_area_'] = hospital_position['metro_area_'].apply(lambda x: x.split(" nonmetropolitan area")[0] if "nonmetropolitan area" in x else x)
hospital_position['metro_match'] = hospital_position['metro_area_'] == hospital_position['city_']
print(hospital_position['metro_match'].value_counts())
'''
False    1325964
True      589868
'''
hospital_position_filtered = hospital_position.loc[
    (hospital_position['city_match'] == 1) | 
    (hospital_position['state_match'] == 1) | 
    (hospital_position['metro_match'] == 1)
]
del hospital_position
#
hospital_position_filtered = hospital_position_filtered[hospital_position_filtered['state']==hospital_position_filtered['State_name']]
hospital_position_filtered.to_csv('Output_data/hospital_position_filtered_full.csv', index=False)
b = hospital_position_filtered[['position_id','company_cleaned','location_raw','state', 'metro_area','Hospital Name','Sate','City','State_name','rcid']]
b.to_csv('Output_data/hospital_position_filtered.csv', index=False)

# %% Make sure that each position is only matched to one hospital
# Count repetitions for each position_id
position_counts = hospital_position_filtered['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# Map the repetition counts back to the original DataFrame
hospital_position_filtered['repetition_count'] = hospital_position_filtered['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_filtered['repetition_count'].value_counts())
'''
1    1148173
2        164
'''
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['city_match'] == 0))]

# Count repetitions for each position_id
position_counts = hospital_position_filtered['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# Map the repetition counts back to the original DataFrame
hospital_position_filtered['repetition_count'] = hospital_position_filtered['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_filtered['repetition_count'].value_counts())
'''
1    1148216
2         12
'''
hospital_position_filtered = hospital_position_filtered[hospital_position_filtered['repetition_count'] != 2] # drop those with uncertain location
hospital_position_filtered = hospital_position_filtered.drop(columns=['repetition_count','Unnamed: 0','location_raw_','city_','location1','state_name_','metro_area_'])
del position_counts
b = hospital_position_filtered[['position_id','company_cleaned','location_raw','state', 'metro_area','Hospital Name','Sate','City','State_name','rcid']]
b.to_csv('Output_data/hospital_position_filtered.csv', index=False)

'''
# %%
# confirm that title_translated has no additional information for title_raw
title_raw_ = hospital_position_filtered['title_raw'].str.lower()
title_raw_ = title_raw_.str.replace(r'[\-]', ' ', regex=True)
title_raw_ = title_raw_.str.replace('/', ' ', regex=True)
title_raw_ = title_raw_.str.replace(r'[().,;]', '', regex=True)
title_raw_ = title_raw_.str.replace(r'[&]', ' and ', regex=True)
title_same = hospital_position_filtered['title_translated'] == title_raw_
print(title_same.value_counts())
True     1110472
False      37744
del title_raw_, title_same
'''

# %% identifying AI employees
print(hospital_position_filtered['description'].count()) # 430979 nonmissing
# %% 
# using AI keywords 
# Step 1: Load AI keywords
ai_keywords = pd.read_excel('Data/AI keywords.xlsx',sheet_name='Final keywords')
ai_keywords = ai_keywords['AI_keywords'].str.lower().tolist()

# Step 2: Search for ai keywords in the title
hospital_position_filtered['title_raw'] = hospital_position_filtered['title_raw'].astype(str).fillna('')
hospital_position_filtered['ai_title'] = hospital_position_filtered['title_raw'].str.lower().apply(lambda x: any(keyword in x for keyword in ai_keywords or x.startswith('ai ') or x.startswith('ml ') or x.startswith('nlp ')))
print(hospital_position_filtered['ai_title'].value_counts())
'''
False    1147913
True         303
'''
# Step 3: Search for ai keywords in the description
hospital_position_filtered['description'] = hospital_position_filtered['description'].astype(str).fillna('')
hospital_position_filtered['ai_description'] = hospital_position_filtered['description'].str.lower().apply(lambda x: any(keyword in x for keyword in ai_keywords or x.startswith('ai ') or x.startswith('ml ') or x.startswith('nlp ')))
print(hospital_position_filtered['ai_description'].value_counts())
'''
False    1146713
True        1503
'''

# &% Generate the data for 430 currect AI positions
print(hospital_position_filtered[pd.isna(hospital_position_filtered['enddate'])]['user_id'].nunique())
print(hospital_position_filtered[(pd.isna(hospital_position_filtered['enddate'])) & 
                                 ((hospital_position_filtered['ai_title'] == 1) | 
                                  (hospital_position_filtered['ai_description'] == 1))
                                ]['user_id'].nunique())
a = hospital_position_filtered[pd.isna(hospital_position_filtered['enddate'])][['ai_title', 'ai_description']].max(axis=1)
a = hospital_position_filtered[(pd.isna(hospital_position_filtered['enddate'])) & 
                                 ((hospital_position_filtered['ai_title'] == 1) | 
                                  (hospital_position_filtered['ai_description'] == 1))]
a.to_excel('Output_data/a.xlsx', index=False)
b = pd.merge(a,hospital_user,on='user_id',how='left')
a_list = a[['user_id']][a['user_id'].notna()]
a_list = a_list.drop_duplicates()
c = pd.merge(a_list,hospital_user_education,on='user_id',how='inner')
d = pd.merge(a_list,hospital_user_skill,on='user_id',how='inner')
with pd.ExcelWriter('Output_data/currect_AI_worker.xlsx') as writer:  # doctest: +SKIP
    b.to_excel(writer, sheet_name='Positions', index=False)
    c.to_excel(writer, sheet_name='AI worker education', index=False)
    d.to_excel(writer, sheet_name='AI worker skill', index=False)
