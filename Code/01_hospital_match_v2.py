# -*- coding: utf-8 -*-
"""
Created by Yanzun Yang
Last edit: 2024/12/11
This file is to match AHA hospitals to all possible companies in Revelio Labs based on names and ownerships. 
"""



# %%
import pandas as pd
import os
import glob
import pyarrow.parquet as pq
import numpy as np
import datetime
from rapidfuzz import process, fuzz
import multiprocessing as mp



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


# %% set up the hospital names for match
# import hospital list data
hospital_list = pd.read_excel('Output_data/hospital_list.xlsx')
# import hospital site names
hospital_site_names = pd.read_excel('Output_data/hospital_site_names.xlsx')[['Hospital Name', 'Site Name']]
hospital_site_names = pd.merge(hospital_list, hospital_site_names, on='Hospital Name', how='inner')
# combine original names and site names
hospital_site_names = hospital_site_names.rename(columns={'Site Name': 'Hospital Name1'})
hospital_list['Hospital Name1'] = hospital_list['Hospital Name']
hospital_list = pd.concat([hospital_list, hospital_site_names], ignore_index=True)
hospital_list = hospital_list.drop_duplicates()
hospital_list.to_excel('Output_data/hospital_list_full.xlsx', index=False)
# manually add Adventist Health System Sunbelt Healthcare Corp to Adventhealth Orlando and Mass General to Massachusetts General Hospital in hospital_list_full


# %% clean the hospital names
# convert to lower case
hospital_list['company'] = hospital_list['Hospital Name1'].str.lower()
# clean the names
hospital_list['company'] = hospital_list['company'].str.replace(r'[,]', '', regex=True)
hospital_list['company'] = hospital_list['company'].str.replace(' inc.', '', case=False)
hospital_list['company'] = hospital_list['company'].str.replace(' llc', '', case=False)
hospital_list['company'] = hospital_list['company'].str.replace(r'[\-\+&]', ' ', regex=True)
hospital_list['company'] = hospital_list['company'].str.replace('   ', ' ')
hospital_names = hospital_list['company'].tolist()
# compile regex pattern for hospitals to avoid looping over individual names
hospital_pattern = '|'.join(re.escape(name) for name in hospital_list['company'])
hospital_regex = re.compile(hospital_pattern)


# %% exact name matching
# import company ref data
# define the folder path containing the files
folder_path = 'F:/academic_company_ref/*.parquet'
# get a list of all parquet files in the folder
file_list = glob.glob(folder_path)
# initialize an empty list to hold each DataFrame
df_hospital_list = []
# loop through each file and append it to the list
i = 0
for file in file_list:
    print(file)
    parquet = pq.read_table(file)
    df = parquet.to_pandas()[['rcid','company']]
    df = df[df['company'].str.match('^[a-zA-Z]')] # only keep whose names start with English letters
    # convert to lower case
    df['company'] = df['company'].str.lower()
    # clean the names
    df['company'] = df['company'].str.replace(r'[,]', '', regex=True)
    df['company'] = df['company'].str.replace(' inc.', '', case=False)
    df['company'] = df['company'].str.replace(' llc', '', case=False)
    df['company'] = df['company'].str.replace(r'[\-\+&]', ' ', regex=True)
    df['company'] = df['company'].str.replace('   ', ' ')
    # filter using the compiled hospital regex
    df_hospital = df[df['company'].str.contains(hospital_regex, regex=True)]
    df_hospital_list.append(df_hospital)
    i += 1
# concatenate all DataFrames in the list
matched_hospitals = pd.concat(df_hospital_list, ignore_index=True)
matched_hospitals.to_excel("Output_data/matched_hospitals.xlsx", index=False)
del parquet, df_hospital, df_hospital_list, df


# %% fuzzy matching
# define the folder path containing the files
folder_path = 'F:/academic_company_ref/*.parquet'
# define the function that processes each company_ref file
def process_file(file):
    parquet = pq.read_table(file)
    df = parquet.to_pandas()[['rcid','company']]
    df = df[df['company'].str.match('^[a-zA-Z]')] # only keep whose names start with English letters
    # convert to lower case
    df['company'] = df['company'].str.lower()
    # clean the names
    df['company'] = df['company'].str.replace(r'[,]', '', regex=True)
    df['company'] = df['company'].str.replace(' inc.', '', case=False)
    df['company'] = df['company'].str.replace(' llc', '', case=False)
    df['company'] = df['company'].str.replace(r'[\-\+&]', ' ', regex=True)
    df['company'] = df['company'].str.replace('   ', ' ')
    # unmatched companies
    unmatched_df = pd.merge(df, matched_hospitals[['rcid']], on='rcid', how='left', indicator=True)
    unmatched_df = unmatched_df[unmatched_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    # 
    fuzzy_matches = []
    # use RapidFuzz to apply fuzzy matching with a threshold 80
    for _, row in unmatched_df.iterrows():
        company_name = row['company']
        match = process.extractOne(company_name, hospital_names, scorer=fuzz.token_sort_ratio, score_cutoff=80)
        
        # if a match is found, add it to the fuzzy matches list
        if match:
            fuzzy_matches.append(row)

    # convert fuzzy_matches to a DataFrame
    fuzzy_matched_df = pd.DataFrame(fuzzy_matches)
    
    return fuzzy_matched_df
 
# get list of all parquet files
file_list = glob.glob(folder_path)

# use multiprocessing to process files in parallel
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
print(ncpus)
pool = mp.Pool(processes=ncpus)   
# process files in parallel and concatenate the results
results = pool.map(process_file, file_list)

# concatenate all results into one DataFrame
fuzzy_matched_hospitals = pd.concat(results, ignore_index=True)

# save the results to Excel
fuzzy_matched_hospitals.to_excel("fuzzy_matched_hospitals.xlsx", index=False)


# %% merge all matched Revelio Labs companies to the hospital lists
# Step 1a: Define a function to match hospitals using str.contains
def str_contains_match(hospital_name, matched_hospitals):
    matches = matched_hospitals[matched_hospitals['company'].str.contains(hospital_name, na=False)]
    if not matches.empty:
        return matches[['company', 'rcid']].to_dict('records') if not matches.empty else []

# Step 1b: Define a function to match hospitals using fuzzy matching
def fuzzy_match(hospital_name, fuzzy_matched_hospitals):
    matches = process.extract(hospital_name, fuzzy_matched_hospitals['company'], scorer=fuzz.token_sort_ratio)
    matched_records = []
    for match in matches:
        matched_company, score = match[0], match[1]
        # Only keep matches with a score above the cutoff
        if score >= 80:
            # Get all IDs corresponding to the matched company name
            matched_rows = fuzzy_matched_hospitals[fuzzy_matched_hospitals['company'] == matched_company]
            if not matched_rows.empty:
                for _, row in matched_rows.iterrows():
                    matched_records.append({'company': matched_company, 'rcid': row['rcid']})
        return matched_records

# Step 3: Apply str.contains() matching and fuzzy matching separately
hospital_list['str_contains_matches'] = hospital_list['company'].apply(lambda x: str_contains_match(x, matched_hospitals))
hospital_list['fuzzy_matches'] = hospital_list['company'].apply(lambda x: fuzzy_match(x, fuzzy_matched_hospitals))
hospital_list['str_contains_matches'] = hospital_list['str_contains_matches'].apply(lambda x: [] if x is None else x)

# Step 4: Combine the results of both matching methods into a single list of matched records
def combine_matches(row):
    str_contains_matches = row['str_contains_matches'] if isinstance(row['str_contains_matches'], list) else []
    fuzzy_matches = row['fuzzy_matches'] if isinstance(row['fuzzy_matches'], list) else []
    combined = str_contains_matches + fuzzy_matches
    return combined if combined else []

hospital_list['combined_matches'] = hospital_list.apply(lambda row: combine_matches(row), axis=1)

# Step 5: Expand the results so that each match is on a separate row
hospital_list_expanded = hospital_list.explode('combined_matches').reset_index(drop=True)
hospital_list_expanded['combined_matches'] = hospital_list_expanded['combined_matches'].apply(
    lambda x: x if isinstance(x, dict) else {}
)

# Step 6: Expand the dictionary in 'combined_matches' to separate columns for 'matched_company' and 'rcid'
hospital_list_expanded[['matched_company', 'rcid']] = pd.json_normalize(hospital_list_expanded['combined_matches'])

# Step 7: Drop the original 'combined_matches' column as we now have separate columns
hospital_list_expanded = hospital_list_expanded.drop(columns=['combined_matches','str_contains_matches','fuzzy_matches'])

# drop duplicates
hospital_list_expanded = hospital_list_expanded.drop_duplicates() # no one dropped

# save
hospital_list_expanded.to_excel("Output_data/hospital_list_final_v2.xlsx", index=False)


# %% add the child and parent companies of the match companies
# get the matched rcid
hospital_list = pd.read_excel('Output_data/hospital_list_final_v2.xlsx')
hospital_list = hospital_list[['rcid']][hospital_list['rcid'].notna()]
hospital_list = hospital_list.drop_duplicates()


# %% first get the child and parent rcid of matched hospitals
# import company ref data
# define the folder path containing the files
folder_path = 'F:/academic_company_ref/*.parquet'
# get a list of all parquet files in the folder
file_list = glob.glob(folder_path)
# initialize an empty list to hold each DataFrame
df_hospital_list = []
# loop through each file and append it to the list
for file in file_list:
    print(file)
    parquet = pq.read_table(file)
    df = parquet.to_pandas()[['rcid','child_rcid','ultimate_parent_rcid']]
    df = pd.merge(df, hospital_list, on='rcid', how='inner')
    df_hospital_list.append(df)
    
# concatenate all DataFrames in the list
hospital_list_ownership = pd.concat(df_hospital_list, ignore_index=True)
del parquet, df_hospital_list, df


# %% combine all rcid
# original rcid
combined_rcid1 = hospital_list_ownership[['rcid']]
combined_rcid1['rcid_'] = combined_rcid1['rcid']
# child rcid
combined_rcid2 = hospital_list_ownership[['rcid','child_rcid']]
combined_rcid2 = combined_rcid2.rename(columns={'child_rcid':'rcid_'})
# parent rcid
combined_rcid3 = hospital_list_ownership[['rcid','ultimate_parent_rcid']]
combined_rcid3 = combined_rcid3.rename(columns={'ultimate_parent_rcid':'rcid_'})
# concatenate
combined_rcid = pd.concat([combined_rcid1,combined_rcid2,combined_rcid3],ignore_index=True)
combined_rcid = combined_rcid.drop_duplicates()
del hospital_list_ownership


# %% merge back with hospitals
hospital_list = pd.read_excel('Output_data/hospital_list_final_v2.xlsx')
#
hospital_list = pd.merge(hospital_list, combined_rcid, on='rcid', how='left', indicator=True)
print(hospital_list['_merge'].value_counts())
#
hospital_list = hospital_list.drop(columns=['rcid','_merge'])
hospital_list = hospital_list.rename(columns={'rcid_':'rcid'})
hospital_list = hospital_list.drop_duplicates()
#
hospital_list.to_csv('Output_data/hospital_list_final_with_network_v2.csv', index=False)
