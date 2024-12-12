# -*- coding: utf-8 -*-
"""
Created by Yanzun Yang
Last edit: 2024/11/18
This file is to match information for workers who have ever worked for hospitals and identify who have ever worked as AI workers
"""

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


# %% obtain the list of hospital workers
hospital_position_filtered = pd.read_parquet('Output_data/hospital_position_filtered_final.parquet')
user_list = hospital_position_filtered[['user_id']][hospital_position_filtered['user_id'].notna()]
user_list = user_list.drop_duplicates()
user_list.to_csv('Output_data/hospital_user_list_v2.csv', index=False)


# %% merge with all historical positions
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
    df = parquet.to_pandas()[['user_id','position_id','rcid','title_raw','title_translated','description']]
    df = pd.merge(df, user_list, on='user_id', how='inner')
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
hospital_user_all_position = pd.concat(df_list, ignore_index=True)
del parquet, df_list, df  


# %% merge with individual user data
# import individual user data
# define the folder path containing the files
folder_path = 'F:/academic_individual_user/*.parquet'
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
    df = pd.merge(df, user_list, on='user_id', how='inner')
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
hospital_user = pd.concat(df_list, ignore_index=True)
hospital_user.to_csv('Output_data/hospital_user_v2.csv', index=False)
del parquet, df_list, df


# %% merge with individual education data
# import individual user's education data
# define the folder path containing the files
folder_path = 'F:/academic_individual_user_education/*.parquet'
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
    df = pd.merge(df, user_list, on='user_id', how='inner')
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
hospital_user_education = pd.concat(df_list, ignore_index=True)
hospital_user_education.to_csv('Output_data/hospital_user_education.csv', index=False)
del parquet, df_list


# %% merge with individual skill data
# import individual user's skill data
# define the folder path containing the files
folder_path = 'F:/academic_individual_user_skill/*.parquet'
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
    df = pd.merge(df, user_list, on='user_id', how='inner')
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
hospital_user_skill = pd.concat(df_list, ignore_index=True)
hospital_user_skill.to_csv('Output_data/hospital_user_skill.csv', index=False)
del parquet, df_list, df


# %% identify AI workers for any position
is_english = hospital_user_all_position['title_raw'].str.contains(r'[a-zA-Z]', regex=True) # only 636 are not
# search for ai keywords in the title
hospital_user_all_position['title_raw'] = hospital_user_all_position['title_raw'].astype(str).fillna('')
hospital_user_all_position['ai_title'] = hospital_user_all_position['title_raw'].str.lower().apply(lambda x: any(keyword in x for keyword in ai_keywords or x.startswith('ai ') or x.startswith('ml ') or x.startswith('nlp ')))
print(hospital_user_all_position['ai_title'].value_counts())

# search for ai keywords in the description
hospital_user_all_position['description'] = hospital_user_all_position['description'].astype(str).fillna('')
hospital_user_all_position['ai_description'] = hospital_user_all_position['description'].str.lower().apply(lambda x: any(keyword in x for keyword in ai_keywords or x.startswith('ai ') or x.startswith('ml ') or x.startswith('nlp ')))
print(hospital_user_all_position['ai_description'].value_counts())

# save
hospital_user_all_position = hospital_user_all_position[['position_id','user_id','rcid','ai_title','ai_description']]
hospital_user_all_position.to_csv('Output_data/hospital_user_all_position_ai_v2.csv', index=False)


# %% generate an indicator for ever-AI-workers
hospital_user_all_position['ai_ind'] = hospital_user_all_position[['ai_title', 'ai_description']].max(axis=1)

# group by user_id and take the maximum of ai_title for each group
hospital_user_ever_ai = hospital_user_all_position.groupby('user_id', as_index=False)['ai_ind'].max()
print(hospital_user_ever_ai['ai_ind'].value_counts())

# save
hospital_user_ever_ai.to_csv('Output_data/hospital_user_ever_ai_v2.csv', index=False)


# %% find AI workers according to Gelles et al. (2024)
# # %%
# # import company ref data
# # define the folder path containing the files
# folder_path = 'F:/academic_company_ref/*.parquet'
# # get a list of all parquet files in the folder
# file_list = glob.glob(folder_path)
# # initialize an empty list to hold each DataFrame
# df_list = []
# # loop through each file and append it to the list
# i = 0
# for file in file_list:
#     print(file)
#     parquet = pq.read_table(file)
#     df = parquet.to_pandas()[['rcid','company']]
#     df = df[df['company'].str.match('^[a-zA-Z]')] # only keep whose names start with English letters
#     df['file'] = i
#     df_list.append(df)
#     i += 1
# # concatenate all DataFrames in the list
# company_ref = pd.concat(df_list, ignore_index=True)
# del parquet, df, df_list


# # %% find 46 AI start-ups in the US
# # import ai start-ups list
# ai_startup_list = pd.read_excel('Data/CB-Insights_AI-100-2021.xlsx',sheet_name='AI 46')

# # %%
# # convert to lower case
# company_ref['company'] = company_ref['company'].str.lower()
# ai_startup_list['company'] = ai_startup_list['Company'].str.lower()
# # clean the names
# company_ref['company'] = company_ref['company'].str.replace(r'[,]', '', regex=True)
# company_ref['company'] = company_ref['company'].str.replace(' inc.', '', case=False)
# company_ref['company'] = company_ref['company'].str.replace(' llc', '', case=False)
# company_ref['company'] = company_ref['company'].str.replace(r'[\-\+&]', ' ', regex=True)
# company_ref['company'] = company_ref['company'].str.replace('   ', ' ')

# # %%
# # match with exact company names 
# merged_df = pd.merge(company_ref, ai_startup_list, on='company', how='right',indicator=True)
# print(merged_df['_merge'].value_counts())
# # merged_df.to_csv('Output_data/merged_df.csv', index=False)
# # keep unmatched
# ai_startup_list1 = merged_df[merged_df['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])

# # %%
# # find those in company_ref that contain unmatched hospitals
# for ai_startup in ai_startup_list1['company']:
#     company_ref.loc[
#         company_ref['company'].str.contains(ai_startup), 'company'
#     ] = ai_startup    
    
# # %%
# # match with companies whose names contain unmatched hospitals
# merged_df1 = pd.merge(company_ref, ai_startup_list1, on='company', how='right',indicator=True)
# print(merged_df1['_merge'].value_counts())

# # %%
# # put all together
# ai_startup_list_final = pd.merge(merged_df, ai_startup_list, on=['company','Company','State'], how='right')
# for i in range(1,2):
#     ai_startup_list_final = pd.merge(eval(f'merged_df{i}'), ai_startup_list_final, on=['company','Company','State'], how='right', suffixes=('_x', '_y'))
#     ai_startup_list_final.loc[ai_startup_list_final['rcid_x'].notna(), 'rcid_y'] = ai_startup_list_final['rcid_x']
#     ai_startup_list_final.loc[ai_startup_list_final['file_x'].notna(), 'file_y'] = ai_startup_list_final['file_x']
#     ai_startup_list_final.loc[ai_startup_list_final['_merge_x'].notna(), '_merge_y'] = ai_startup_list_final['_merge_x']
#     ai_startup_list_final = ai_startup_list_final.drop(columns=['rcid_x','file_x','_merge_x'])
#     ai_startup_list_final = ai_startup_list_final.rename(columns={'rcid_y': 'rcid','file_y': 'file','_merge_y': '_merge'})

# ai_startup_list_final.to_excel('Output_data/ai_startup_list_final.xlsx', index=False)
# del company_ref

# after matching AI startups to Revelio Labs companies
ai_startup_list_final = pd.read_excel('Output_data/ai_startup_list_final.xlsx')
ai_startup_list = ai_startup_list_final[['rcid']][ai_startup_list_final['rcid'].notna()]
ai_startup_list = ai_startup_list.drop_duplicates()


# %% find positions in ai start-ups
hospital_user_ai_startup_position = pd.merge(hospital_user_all_position, ai_startup_list, on=['rcid'],how='inner')
hospital_user_ai_startup_position = hospital_user_ai_startup_position[['position_id']][hospital_user_ai_startup_position['position_id'].notna()]
hospital_user_ai_startup_position = hospital_user_ai_startup_position.drop_duplicates()


# %% match to titles for positions in ai start-ups
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
    df = pd.merge(df, hospital_user_ai_startup_position, on='position_id', how='inner')
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
hospital_user_ai_startup_positions = pd.concat(df_list, ignore_index=True)
del parquet, df_list


# %% filter the positions in ai start-ups according to locations
hospital_user_ai_startup_position_full = pd.merge(hospital_user_ai_startup_positions, ai_startup_list_final, on='rcid', how='inner')
# check positions
hospital_user_ai_startup_position_full['state_match'] = hospital_user_ai_startup_position_full['state'] == hospital_user_ai_startup_position_full['State']
print(hospital_user_ai_startup_position_full['state_match'].value_counts())
hospital_user_ai_startup_position_full.to_excel('Output_data/hospital_user_ai_startup_position_full_v2.xlsx', index=False)
hospital_user_ai_startup_position_full = hospital_user_ai_startup_position_full[hospital_user_ai_startup_position_full['state_match']==1]


# %% identify AI jobs
# using AI job titles
# Step 1: Load AI job titles
ai_titles = pd.read_excel('Data/AI keywords.xlsx',sheet_name='compute survey')
ai_titles = ai_titles['AI_titles'].str.lower().tolist()

# Step 2: Search for ai keywords in the title
hospital_user_ai_startup_position_full['title_raw'] = hospital_user_ai_startup_position_full['title_raw'].astype(str).fillna('')
hospital_user_ai_startup_position_full['ai_title_startup'] = hospital_user_ai_startup_position_full['title_raw'].str.lower().apply(lambda x: any(title == x for title in ai_titles))
print(hospital_user_ai_startup_position_full['ai_title_startup'].value_counts())


# %% finalizing AI workers identification
# load ever-AI indicators from all historial positions
hospital_user_ever_ai = pd.read_csv('Output_data/hospital_user_ever_ai_v2.csv')
# group by user_id and take the maximum of ai_title for each group
hospital_user_ever_ai1 = hospital_user_ai_startup_position_full.groupby('user_id', as_index=False)['ai_title_startup'].max()
print(hospital_user_ever_ai1['ai_title_startup'].value_counts())
hospital_user_ever_ai1.to_csv('Output_data/hospital_user_ever_ai1_v2.csv', index=False)

# aggregate two ever-AI indicators
hospital_user_ever_ai = pd.merge(hospital_user_ever_ai, hospital_user_ever_ai1, on='user_id', how='left')
hospital_user_ever_ai['ai_title_startup'] = hospital_user_ever_ai['ai_title_startup'].fillna(False)
hospital_user_ever_ai.to_csv('Output_data/hospital_user_ever_ai_final.csv', index=False)
print(hospital_user_ever_ai[hospital_user_ever_ai['ai_title_startup']])

