# -*- coding: utf-8 -*-
"""
Created by Yanzun Yang
Last edit: 2024/12/12
This file is to match with hospital positions and identify AI positions.
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
import openai
import time
import re

"""
"""

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


# %% get the rcids for hospitals
hospital_list = pd.read_csv('Output_data/hospital_list_final_with_network_v2.csv')
hospital_list = hospital_list[['rcid']][hospital_list['rcid'].notna()]
hospital_list = hospital_list.drop_duplicates()

# %% merge with positions
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


# %% a little clean
hospital_position.to_csv('F:/output_data/hospital_position_original_v2.csv')
hospital_position = hospital_position.drop(columns=['company_raw','company_linkedin_url','company_name','ultimate_parent_rcid','ultimate_parent_company_name','naics_code','naics_description'])
print(hospital_position['state'][hospital_position['country']=='empty'].value_counts()) # as long as country is empty, state is also empty
hospital_position = hospital_position[hospital_position['country']=='United States']


# %% merge with hospitals
hospital_list = pd.read_csv('Output_data/hospital_list_final_with_network_v2.csv')
hospital_list = hospital_list[['rcid','Hospital Name','Sate','City']][hospital_list['rcid'].notna()]
# merge state abbr. to full names
US_states = pd.read_excel('Data/US_states.xlsx')
hospital_list = pd.merge(US_states, hospital_list, on='Sate', how='right')
# merge positons with hospitals through rcids
hospital_position = pd.merge(hospital_position, hospital_list, on='rcid', how='inner') # 4656 hospitals have position data


# %% match companies based on locations
# city match flags
hospital_position['location_raw_'] = hospital_position['location_raw'].str.lower()
hospital_position['city_'] = hospital_position['City'].str.lower()
hospital_position['city_match'] = hospital_position.apply(lambda row: row['city_'] in row['location_raw_'], axis=1)
print(hospital_position['city_match'].value_counts())

# state match flags when position location doesn't include cities
hospital_position['location1'] = hospital_position['location_raw'].apply(lambda x: x.split(',')[0] if ',' in x else x)
hospital_position['location1'] = hospital_position['location1'].str.lower()
hospital_position['state_name_'] = hospital_position['State_name'].str.lower()
hospital_position['state_match'] = hospital_position.apply(lambda row: row['state_name_'] in row['location1'], axis=1)
print(hospital_position['state_match'].value_counts())

# metropolitan area in positions matched to hospital cities flags
hospital_position['metro_area_'] = hospital_position['metro_area'].apply(lambda x: x.split(" metropolitan area")[0] if "metropolitan area" in x else x)
hospital_position['metro_area_'] = hospital_position['metro_area_'].apply(lambda x: x.split(" nonmetropolitan area")[0] if "nonmetropolitan area" in x else x)
hospital_position['metro_match'] = hospital_position['metro_area_'] == hospital_position['city_']
print(hospital_position['metro_match'].value_counts())

# keep either city matched or metro matched
hospital_position_filtered = hospital_position.loc[
    (hospital_position['city_match'] == 1) |  
    (hospital_position['metro_match'] == 1)
]
del hospital_position


# %% clean filtered hospital positions
# hospital_position_filtered = pd.read_parquet('Output_data/hospital_position_filtered_1106.parquet')
# make sure state matchced
hospital_position_filtered = hospital_position_filtered[hospital_position_filtered['state']==hospital_position_filtered['State_name']] 
# drop duplicates
hospital_position_filtered = hospital_position_filtered.drop_duplicates() 
hospital_position_filtered.to_csv('Output_data/hospital_position_filtered_full_v2.csv', index=False)


# %% make sure that each position is only matched to one hospital
# count repetitions for each position_id
position_counts = hospital_position_filtered['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# map the repetition counts back to the original DataFrame
hospital_position_filtered['repetition_count'] = hospital_position_filtered['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_filtered['repetition_count'].value_counts())

# multiple matches indicator
hospital_position_filtered['multiple_matches'] = hospital_position_filtered['repetition_count'] > 1

# keep the closest hospital that matches on cities if it exists
hospital_position_filtered['city_match_max'] = hospital_position_filtered.groupby('position_id')['city_match'].transform('max')
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] > 1) & (hospital_position_filtered['city_match'] == 0) & (hospital_position_filtered['city_match_max'] == 1))] # 3694593 obs
hospital_position_filtered = hospital_position_filtered.drop(columns=['city_match_max'])

# count repetitions for each position_id
position_counts = hospital_position_filtered['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# map the repetition counts back to the original DataFrame
hospital_position_filtered['repetition_count'] = hospital_position_filtered['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_filtered['repetition_count'].value_counts())

# keep position and hospital to manually check multiple hospital matches
b = hospital_position_filtered[['position_id','company_cleaned','location_raw','state', 'metro_area','Hospital Name','Sate','City','State_name','rcid','city_match','metro_match']]
b.to_csv('Output_data/hospital_position_filtered_v2.csv', index=False)


# %% further filters
# match with the hospital with the same company name
hospital_position_filtered['same_name'] = hospital_position_filtered['company_cleaned'].str.lower() == hospital_position_filtered['Hospital Name'].str.lower()
hospital_position_filtered['same_name_max'] = hospital_position_filtered.groupby('position_id')['same_name'].transform('max')
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] > 1) & (hospital_position_filtered['same_name'] == 0) & (hospital_position_filtered['same_name_max'] == 1))]
hospital_position_filtered = hospital_position_filtered.drop(columns=['same_name','same_name_max'])

#
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] > 1) & (hospital_position_filtered['City'] == 'Little Rock') & (hospital_position_filtered['location_raw'].str.contains('North Little Rock')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['City'] == 'Sun City') & (hospital_position_filtered['location_raw'].str.contains('Sun City West')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['City'] == 'San Francisco') & (hospital_position_filtered['location_raw'].str.contains('South San Francisco')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['City'] == 'Chicago') & (hospital_position_filtered['location_raw'].str.contains('North Chicago')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['City'] == 'Houston') & (hospital_position_filtered['location_raw'].str.contains('Fort Sam Houston')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['City'] == 'Spokane') & (hospital_position_filtered['location_raw'].str.contains('Spokane Valley')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Scripps Memorial Hospital-La Jolla') & (hospital_position_filtered['location_raw'].str.contains('Green')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Loma Linda University Medical Center') & (hospital_position_filtered['company_cleaned'].str.contains('murrieta')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Brigham and Women\'s Faulkner Hospital'))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] > 1) & ~(hospital_position_filtered['Hospital Name'].str.contains('McKeesport')) & (hospital_position_filtered['location_raw'].str.contains('mckeesport')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Baptist Health Paducah'))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Dallas Medical Center') & (hospital_position_filtered['company_cleaned'].str.contains('methodist')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Texas Health Presbyterian Hospital Dallas') & (hospital_position_filtered['company_cleaned'].str.contains('texas institute for surgery at texas health presbyterian dallas')))] 
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Texas Health Presbyterian Hospital Dallas') & (hospital_position_filtered['company_cleaned'].str.contains('veterans affairs north texas health care system')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'] == 'Park Plaza Hospital'))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 6) & (hospital_position_filtered['Hospital Name'] != 'Veterans Affairs North Texas Health Care System') & (hospital_position_filtered['company_cleaned'].str.contains('veterans affairs north texas health care system')))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] > 1) & (hospital_position_filtered['City'] == 'New York') & (hospital_position_filtered['location1'] != "new york") & ~(hospital_position_filtered['location_raw'].str.contains(r"[\/&]")))]
hospital_position_filtered = hospital_position_filtered[~((hospital_position_filtered['repetition_count'] == 2) & (hospital_position_filtered['Hospital Name'].str.contains('Mary')) & (hospital_position_filtered['company_cleaned'].str.contains('joseph')))]


# %% keep the biggest/flagship hospital
# load the file that manually picked the flagship hospital
position_multiple_matches_hospitals = pd.read_excel('Output_data/position_multiple_matches_hospitals.xlsx')
hospital_position_filtered = pd.merge(hospital_position_filtered, position_multiple_matches_hospitals, on=['Hospital Name','Sate','City','repetition_count'], how='left', indicator=True)
print(hospital_position_filtered['_merge'].value_counts())

# keep the flagship hospital
hospital_position_filtered.loc[hospital_position_filtered['Hospital_Name_Flag'].notna(), 'Hospital Name'] = hospital_position_filtered['Hospital_Name_Flag']
hospital_position_filtered.loc[hospital_position_filtered['City_Flag'].notna(), 'City'] = hospital_position_filtered['City_Flag']
hospital_position_filtered = hospital_position_filtered.drop(columns=['Hospital_Name_Flag','City_Flag','Sate_Flag','_merge','city_','city_match','metro_match'])
hospital_position_filtered = hospital_position_filtered.drop_duplicates() 
hospital_position_filtered = hospital_position_filtered.drop(columns=['repetition_count'])

# find duplicates
# count repetitions for each position_id
position_counts = hospital_position_filtered['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# map the repetition counts back to the original DataFrame
hospital_position_filtered['repetition_count'] = hospital_position_filtered['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_filtered['repetition_count'].value_counts())

# keep the left hospital position duplicates
hospital_position_duplicates = hospital_position_filtered[hospital_position_filtered['repetition_count']>1]
hospital_position_duplicates = hospital_position_duplicates['position_id']
hospital_position_duplicates = pd.merge(hospital_position_duplicates, b, on='position_id', how='left')
hospital_position_duplicates.to_excel('Output_data/hospital_position_duplicates.xlsx', index=False)


# %% track the number of beds for hospitals with position duplicates
hospital_position_duplicates_h = hospital_position_duplicates[['Hospital Name', 'Sate', 'City']]
hospital_position_duplicates_h = hospital_position_duplicates_h.drop_duplicates()

# set up the API key
openai.api_key = ''

# add an empty column 'number_of_beds' to store the responses
hospital_position_duplicates_h['number_of_beds'] = None

# loop over each row in the DataFrame
for index, row in hospital_position_duplicates_h.iterrows():
    # construct the prompt
    prompt = f"Provide the number of beds for {row['Hospital Name']} located in {row['City']}, {row['Sate']}."

    try:
        # make the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        # extract the response content
        bed_count = response['choices'][0]['message']['content'].strip()
        hospital_position_duplicates_h.at[index, 'number_of_beds'] = bed_count

        # delay to prevent hitting rate limits
        time.sleep(1)

    except Exception as e:
        hospital_position_duplicates_h.at[index, 'number_of_beds'] = 'Error fetching data'
        print(f"Error for hospital {row['Hospital Name']}: {e}")
        
# function to extract only the number from the text
def extract_number(text):
    # remove commas from the text
    text = text.replace(',', '')
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())  # Convert the extracted number to integer
    return None

# apply the function to the 'number_of_beds' column
hospital_position_duplicates_h['number_of_beds_extracted'] = hospital_position_duplicates_h['number_of_beds'].apply(extract_number) # has one NA value
hospital_position_duplicates_h = hospital_position_duplicates_h.drop(columns=['number_of_beds'])
hospital_position_duplicates_h.to_excel('Output_data/hospital_position_duplicates_h.xlsx', index=False)


# %% match back with hospital_position_duplicates and keep the biggest hospital
hospital_position_duplicates = pd.merge(hospital_position_duplicates, hospital_position_duplicates_h, on=['Hospital Name','Sate','City'],how='inner')
hospital_position_duplicates = hospital_position_duplicates.drop_duplicates()

# keep the matched hospital with the highest number of beds
hospital_position_duplicates['number_of_beds_max'] = hospital_position_duplicates.groupby('position_id')['number_of_beds_extracted'].transform('max')
hospital_position_duplicates = hospital_position_duplicates[hospital_position_duplicates['number_of_beds_extracted'] == hospital_position_duplicates['number_of_beds_max']]

# when the number of beds is the same, keep the first hospital
# count repetitions for each position_id
position_counts = hospital_position_duplicates['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# map the repetition counts back to the original DataFrame
hospital_position_duplicates['repetition_count'] = hospital_position_duplicates['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_duplicates['repetition_count'].value_counts())

#
hospital_position_duplicates = hospital_position_duplicates[
    (hospital_position_duplicates['repetition_count'] == 1) | 
    ((hospital_position_duplicates['repetition_count'] > 1) & 
    (hospital_position_duplicates.duplicated(subset='position_id', keep='first') == False))]
 
#
hospital_position_duplicates = hospital_position_duplicates[['position_id','Hospital Name','Sate','City']]
hospital_position_duplicates = hospital_position_duplicates.rename(columns={'Hospital Name':'Hospital_Name_Flag','Sate':'Sate_Flag','City':'City_Flag'})
hospital_position_duplicates = hospital_position_duplicates.drop_duplicates()


# %% match back to hospital_position_filtered
hospital_position_filtered = pd.merge(hospital_position_filtered, hospital_position_duplicates, on=['position_id'], how='left', indicator=True)
print(hospital_position_filtered['_merge'].value_counts())

#
hospital_position_filtered.loc[hospital_position_filtered['Hospital_Name_Flag'].notna(), 'Hospital Name'] = hospital_position_filtered['Hospital_Name_Flag']
hospital_position_filtered.loc[hospital_position_filtered['City_Flag'].notna(), 'City'] = hospital_position_filtered['City_Flag']
hospital_position_filtered = hospital_position_filtered.drop(columns=['Hospital_Name_Flag','City_Flag','Sate_Flag','_merge'])
hospital_position_filtered = hospital_position_filtered.drop_duplicates() 
hospital_position_filtered = hospital_position_filtered.drop(columns=['repetition_count'])

# find duplicates
# count repetitions for each position_id
position_counts = hospital_position_filtered['position_id'].value_counts().reset_index()
position_counts.columns = ['position_id', 'repetition_count']

# map the repetition counts back to the original DataFrame
hospital_position_filtered['repetition_count'] = hospital_position_filtered['position_id'].map(position_counts.set_index('position_id')['repetition_count'])
print(hospital_position_filtered['repetition_count'].value_counts())

# recreate city_match and metro_match
hospital_position_filtered['city_'] = hospital_position_filtered['City'].str.lower()
hospital_position_filtered['city_match'] = hospital_position_filtered.apply(lambda row: row['city_'] in row['location_raw_'], axis=1)
print(hospital_position_filtered['city_match'].value_counts())

hospital_position_filtered['metro_match'] = hospital_position_filtered['metro_area_'] == hospital_position_filtered['city_']
print(hospital_position_filtered['metro_match'].value_counts())

#
hospital_position_filtered = hospital_position_filtered.drop(columns=['repetition_count','location_raw_','city_','location1','state_name_','metro_area_'])
del position_counts
b = hospital_position_filtered[['position_id','company_cleaned','location_raw','state', 'metro_area','Hospital Name','Sate','City','State_name','rcid']]
b.to_csv('Output_data/hospital_position_filtered_v2.csv', index=False)


# %% check false positives according to company names
hospital_position_filtered_rcid = hospital_position_filtered[['rcid']].drop_duplicates()
# match with hospital_list to get the company name
hospital_list = pd.read_csv('Output_data/hospital_list_final_with_network_v2.csv')
hospital_list = hospital_list[['rcid','matched_company']][hospital_list['rcid'].notna()].drop_duplicates()
hospital_position_filtered_rcid = pd.merge(hospital_position_filtered_rcid,hospital_list,on='rcid',how='inner')
# check if the company is a hospital or not
hospital_position_filtered_rcid['is_hospital'] = hospital_position_filtered_rcid['matched_company'].str.contains('hospital|medical|cardiovascular|health|psychiatric|rehabilitation|surgery|treatment|clinic|medicine|recovery|disease|surgical|care|doctor|medcenter|cancer', case=False, regex=True)
print(hospital_position_filtered_rcid['is_hospital'].value_counts())
# keep those that don't include hospital-related keywords
hospital_position_filtered_rcid1 = hospital_position_filtered_rcid[hospital_position_filtered_rcid['is_hospital']==0]

# use ChatGPT to make further judgement
# create an empty column to store whether the company is a hospital or not
hospital_position_filtered_rcid1['is_hospital'] = None

# loop over each row to determine if the company is a hospital
for index, row in hospital_position_filtered_rcid1.iterrows():
    # construct the prompt
    prompt = f"Is company '{row['matched_company']}' a hospital, or related to health/medical care? Please respond with only 'Yes' or 'No'."

    try:
        # make the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,  # Limit the response to ensure it's concise
            temperature=0  # Make the response deterministic
        )

        # extract the response content
        answer = response['choices'][0]['message']['content'].strip()
        hospital_position_filtered_rcid1.at[index, 'is_hospital'] = answer

        # delay to prevent hitting rate limits
        time.sleep(1)

    except Exception as e:
        hospital_position_filtered_rcid1.at[index, 'is_hospital'] = 'Error fetching data'
        print(f"Error for company {row['matched_company']}: {e}")

#
print(hospital_position_filtered_rcid1['is_hospital'].value_counts())
hospital_position_filtered_rcid1.to_excel('Output_data/hospital_position_filtered_rcid.xlsx', index=False)
hospital_position_filtered_rcid1['is_hospital'] = hospital_position_filtered_rcid1['is_hospital'].replace('No.', 'No')
# check univerties/schools
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'the university of alabama at birmingham', 'is_hospital'] = 'No'
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'loma linda university', 'is_hospital'] = 'No'
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'thomas jefferson university', 'is_hospital'] = 'No'
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'phillips school of nursing at mount sinai beth israel', 'is_hospital'] = 'No'
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'upmc shadyside school of nursing', 'is_hospital'] = 'No'
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'avera mckennan school of radiologic technology', 'is_hospital'] = 'No'
hospital_position_filtered_rcid1.loc[hospital_position_filtered_rcid1['matched_company'] == 'upmc hamot school of anesthesia', 'is_hospital'] = 'No'

# match back to hospital_position_filtered
hospital_position_filtered_rcid1 = hospital_position_filtered_rcid1[hospital_position_filtered_rcid1['is_hospital']=='No']
hospital_position_filtered_rcid1 = hospital_position_filtered_rcid1[['rcid']].drop_duplicates()
hospital_position_filtered = pd.merge(hospital_position_filtered, hospital_position_filtered_rcid1, on='rcid', how='left', indicator=True)
print(hospital_position_filtered['_merge'].value_counts())

# drop companies that are not hospitals and get the final version of hospital_position_filtered
hospital_position_filtered = hospital_position_filtered[hospital_position_filtered['_merge']=='left_only'] 
b = hospital_position_filtered[['position_id','company_cleaned','location_raw','state', 'metro_area','Hospital Name','Sate','City','State_name','rcid']]
b.to_csv('Output_data/hospital_position_filtered_v2.csv', index=False)
del b


# %% identifying AI employees
print(hospital_position_filtered['description'].count()) 
# %% 
# using AI keywords 
# Step 1: Load AI keywords
ai_keywords = pd.read_excel('Data/AI keywords.xlsx',sheet_name='Final keywords')
ai_keywords = ai_keywords['AI_keywords'].str.lower().tolist()

# Step 2: Search for ai keywords in the title
hospital_position_filtered['title_raw'] = hospital_position_filtered['title_raw'].astype(str).fillna('')
hospital_position_filtered['ai_title'] = hospital_position_filtered['title_raw'].str.lower().apply(lambda x: any(keyword in x for keyword in ai_keywords or x.startswith('ai ') or x.startswith('ml ') or x.startswith('nlp ')))
print(hospital_position_filtered['ai_title'].value_counts())


# Step 3: Search for ai keywords in the description
hospital_position_filtered['description'] = hospital_position_filtered['description'].astype(str).fillna('')
hospital_position_filtered['ai_description'] = hospital_position_filtered['description'].str.lower().apply(lambda x: any(keyword in x for keyword in ai_keywords or x.startswith('ai ') or x.startswith('ml ') or x.startswith('nlp ')))
print(hospital_position_filtered['ai_description'].value_counts())

#
hospital_position_filtered.to_parquet('F:/output_data/hospital_position_filtered_final.parquet', engine='pyarrow')

