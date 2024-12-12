# -*- coding: utf-8 -*-
"""
Created by Yanzun Yang
Last edit: 2024/12/12
This file is to create final panel datasets
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


# %% load position and ever-AI data
hospital_position_filtered = pd.read_parquet('F:/output_data/hospital_position_filtered_final.parquet')
hospital_user_ever_ai = pd.read_csv('Output_data/hospital_user_ever_ai_final.csv', index=False)



# %% create hospital-level statistics
# generate indicator for AI jobs
hospital_position_filtered_sub = hospital_position_filtered[['user_id','position_id','startdate','enddate','rcid','ai_title','ai_description','Hospital Name','City','Sate']]
hospital_position_filtered_sub['ai_ind_hosp'] = hospital_position_filtered_sub[['ai_title', 'ai_description']].max(axis=1)
hospital_position_filtered_sub = hospital_position_filtered_sub.drop(columns=['ai_title','ai_description'])
print(hospital_position_filtered_sub['ai_ind_hosp'].value_counts())
'''
False    3092514
True        5550
'''
# have ever been AI workers
hospital_position_filtered_sub = pd.merge(hospital_position_filtered_sub, hospital_user_ever_ai, on='user_id', how='inner')
hospital_position_filtered_sub['ever_ai'] = hospital_position_filtered_sub[['ai_ind', 'ai_title_startup']].max(axis=1)
hospital_position_filtered_sub = hospital_position_filtered_sub.drop(columns=['ai_ind','ai_title_startup'])
print(hospital_position_filtered_sub['ever_ai'].value_counts())
'''
False    3080258
True       17806
'''

# is current worker
hospital_position_filtered_sub['current_worker'] = pd.isna(hospital_position_filtered_sub['enddate'])

### aggregate to employee-hospital level
hospital_user_filtered_sub = hospital_position_filtered_sub.groupby(['user_id', 'Hospital Name', 'City', 'Sate']).agg(
    ai_ind_hosp=('ai_ind_hosp', 'max'),
    ever_ai=('ever_ai', 'max'),
    current_worker=('current_worker', 'max')
).reset_index()

### aggregate to hospital level
# for all employees
hospital_all_worker = hospital_user_filtered_sub.groupby(['Hospital Name', 'City', 'Sate']).agg(
    total_worker=('user_id', 'size'),
    total_ai_worker=('ai_ind_hosp', lambda x: (x == 1).sum()),
    total_ever_ai_worker=('ever_ai', lambda x: (x == 1).sum())
).reset_index()
# for current employees
hospital_current_user_filtered_sub = hospital_user_filtered_sub[hospital_user_filtered_sub['current_worker']==1]
hospital_current_worker = hospital_current_user_filtered_sub.groupby(['Hospital Name', 'City', 'Sate']).agg(
    total_worker_current=('user_id', 'size'),
    total_ai_worker_current=('ai_ind_hosp', lambda x: (x == 1).sum()),
    total_ever_ai_worker_current=('ever_ai', lambda x: (x == 1).sum())
).reset_index()
#
hospital_stats = pd.merge(hospital_all_worker, hospital_current_worker, on=['Hospital Name', 'City', 'Sate'], how='left')
hospital_stats['share_ai_worker'] = hospital_stats['total_ai_worker']/hospital_stats['total_worker']
hospital_stats['share_ever_ai_worker'] = hospital_stats['total_ever_ai_worker']/hospital_stats['total_worker']
hospital_stats['share_ai_worker_current'] = hospital_stats['total_ai_worker_current']/hospital_stats['total_worker_current']
hospital_stats['share_ever_ai_worker_current'] = hospital_stats['total_ever_ai_worker_current']/hospital_stats['total_worker_current']
# merge with the initial hospital list
# import hospital list data
hospital_list = pd.read_excel('Data/ListHospitals_RA_Jun282024.xlsx')
# drop those in territories and freely associated state
# list of U.S. territories and freely associated state to exclude
territories = ['PR', 'GU', 'VI', 'AS', 'MP', 'MH']
# filter out rows where the state is one of the territories and freely associated state
hospital_list = hospital_list[~hospital_list['Sate'].isin(territories)]
# drop duplicates
hospital_list = hospital_list.drop_duplicates()
#
hospital_list = pd.merge(hospital_list, hospital_stats, on=['Hospital Name', 'City', 'Sate'], how='left', indicator=True)
hospital_list = hospital_list.rename(columns={'_merge':'matched'})
hospital_list.to_excel('Output_data/hospital_stats_final.xlsx', index=False)



# %% create hospital-year panel
# %% raw data
hospital_position_filtered_sub = hospital_position_filtered.drop(columns=['company_cleaned','location_raw','region','country','state','metro_area','ticker','exchange','cusip','ultimate_parent_factset_name','file','state_match'])
hospital_position_filtered_sub['ai_ind_hosp'] = hospital_position_filtered_sub[['ai_title', 'ai_description']].max(axis=1)
# have ever been AI workers
hospital_position_filtered_sub = pd.merge(hospital_position_filtered_sub, hospital_user_ever_ai, on='user_id', how='inner')
hospital_position_filtered_sub['ever_ai'] = hospital_position_filtered_sub[['ai_ind', 'ai_title_startup']].max(axis=1)
hospital_position_filtered_sub = hospital_position_filtered_sub.drop(columns=['ai_ind','ai_title_startup'])
# aggregate rcid
hospital_position_filtered_sub['aggregate_id'], unique_combinations = pd.factorize(hospital_position_filtered_sub[['Hospital Name', 'Sate', 'City']].apply(tuple, axis=1))
hospital_position_filtered_sub.to_parquet('F:/output_data/hospital_positions_raw_final.parquet', engine='pyarrow')

# %% hospital-year summary statistics
hospital_position_filtered_sub1 = hospital_position_filtered_sub[['user_id','position_id','startdate','enddate','aggregate_id','ai_title','ai_description','Hospital Name','City','Sate','ai_ind_hosp','ever_ai']]
# drop if start date is missing
hospital_position_filtered_sub1 = hospital_position_filtered_sub1.dropna(subset=['startdate'])
# startyear and endyear
hospital_position_filtered_sub1['startdate'] = pd.to_datetime(hospital_position_filtered_sub1['startdate'], errors='coerce')
hospital_position_filtered_sub1['enddate'] = pd.to_datetime(hospital_position_filtered_sub1['enddate'], errors='coerce')
hospital_position_filtered_sub1['startyear'] = hospital_position_filtered_sub1['startdate'].dt.year
hospital_position_filtered_sub1['endyear'] = hospital_position_filtered_sub1['enddate'].dt.year
hospital_position_filtered_sub1['endyear'] = hospital_position_filtered_sub1['endyear'].fillna(2024).astype(int)

# create an empty list to store results
result = []

# iterate over each unique hospital
for hospital in hospital_position_filtered_sub1['aggregate_id'].unique():
    hospital_data = hospital_position_filtered_sub1[hospital_position_filtered_sub1['aggregate_id'] == hospital]
    # iterate over each year in the range
    for year in range(2011,2024):
        # count employees whose employment period includes the current year
        count = hospital_data[(hospital_data['startyear'] <= year) & (hospital_data['endyear'] >= year)]['user_id'].nunique()
        # count employees whose employment period includes the current year and are AI employees
        count_ai = hospital_data[(hospital_data['startyear'] <= year) & (hospital_data['endyear'] >= year) & (hospital_data['ai_ind_hosp'] == 1)]['user_id'].nunique()
        # count employees whose employment period includes the current year and are ever AI employees
        count_ever_ai = hospital_data[(hospital_data['startyear'] <= year) & (hospital_data['endyear'] >= year) & (hospital_data['ever_ai'] == 1)]['user_id'].nunique()
        # append the result
        result.append({'aggregate_id': hospital, 'year': year, 'num_employees': count, 'number_ai_employees': count_ai, 'number_ever_ai_employees': count_ever_ai})

# convert the result list to a DataFrame
hospital_year_panel = pd.DataFrame(result)

# merge with hospital names
hospital_id = hospital_position_filtered_sub1[['Hospital Name','Sate','City','aggregate_id']]
hospital_id = hospital_id.drop_duplicates()
hospital_year_panel = pd.merge(hospital_year_panel,hospital_id,on='aggregate_id',how='inner',indicator=True)
hospital_year_panel = hospital_year_panel.drop(columns=['_merge'])
hospital_year_panel.to_excel('Output_data/hospital_year_panel_final.xlsx', index=False)

