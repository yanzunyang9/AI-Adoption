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


# %%
# import company ref data
# define the folder path containing the files
folder_path = 'F:/academic_company_ref/*.parquet'
# get a list of all parquet files in the folder
file_list = glob.glob(folder_path)
# initialize an empty list to hold each DataFrame
df_list = []
# loop through each file and append it to the list
i = 0
for file in file_list:
    print(file)
    parquet = pq.read_table(file)
    df = parquet.to_pandas()[['rcid','company']]
    df = df[df['company'].str.match('^[a-zA-Z]')] # only keep whose names start with English letters
    df['file'] = i
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
company_ref = pd.concat(df_list, ignore_index=True)
del parquet, df, df_list
'''
parquet = pq.read_table('F:/academic_company_ref/company_ref_0008_part_00.parquet')
df = parquet.to_pandas()
del parquet
'''

# %%
# import hospital list data
hospital_list = pd.read_excel('Data/ListHospitals_RA_Jun282024.xlsx')
# drop those in territories and freely associated state
# list of U.S. territories and freely associated state to exclude
territories = ['PR', 'GU', 'VI', 'AS', 'MP', 'MH']
# filter out rows where the state is one of the territories and freely associated state
hospital_list = hospital_list[~hospital_list['Sate'].isin(territories)]
# drop duplicates
hospital_list = hospital_list.drop_duplicates()

# %%
# convert to lower case
company_ref['company'] = company_ref['company'].str.lower()
hospital_list['company'] = hospital_list['Hospital Name'].str.lower()
# clean the names
company_ref['company'] = company_ref['company'].str.replace(r'[,]', '', regex=True)
hospital_list['company'] = hospital_list['company'].str.replace(r'[,]', '', regex=True)
company_ref['company'] = company_ref['company'].str.replace(' inc.', '', case=False)
hospital_list['company'] = hospital_list['company'].str.replace(' inc.', '', case=False)
company_ref['company'] = company_ref['company'].str.replace(' llc', '', case=False)
hospital_list['company'] = hospital_list['company'].str.replace(' llc', '', case=False)
company_ref['company'] = company_ref['company'].str.replace(r'[\-\+&]', ' ', regex=True)
hospital_list['company'] = hospital_list['company'].str.replace(r'[\-\+&]', ' ', regex=True)
company_ref['company'] = company_ref['company'].str.replace('   ', ' ')
hospital_list['company'] = hospital_list['company'].str.replace('   ', ' ')
# a subset of companies who are more likely to match with hospitals
company_ref_filtered = company_ref[company_ref['company'].str.contains('hospital|medical|cardiovascular|health|psychiatric|rehabilitation|surgery|treatment|clinic|medicine|recovery|disease|surgical|care|doctor|medcenter|cancer', case=False, regex=True)]
# save
company_ref.to_csv('Output_data/company_ref.csv', index=False)
company_ref_filtered.to_csv('Output_data/company_ref_filtered.csv', index=False)

# %% Step 1
# match with exact company names 
merged_df = pd.merge(company_ref, hospital_list, on='company', how='right',indicator=True)
print(merged_df['_merge'].value_counts())
merged_df.to_csv('Output_data/merged_df.csv', index=False)
# keep unmatched
hospital_list1 = merged_df[merged_df['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])

# %% Step 2
# find those in filtered company_ref that contain unmatched hospitals
for hospital in hospital_list1['company']:
    company_ref_filtered.loc[
        company_ref_filtered['company'].str.contains(hospital), 'company'
    ] = hospital    
    
# %%
# match with companies whose names contain unmatched hospitals
merged_df1 = pd.merge(company_ref_filtered, hospital_list1, on='company', how='right',indicator=True)
print(merged_df1['_merge'].value_counts())
merged_df1.to_csv('Output_data/merged_df1.csv', index=False)
# keep unmatched
hospital_list2 = merged_df1[merged_df1['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])

'''
# %% find the google map name for unmatched hospitals
# Load the list of unmatched hospital names
hospitals_for_gm = merged_df1[merged_df1['_merge'] == 'right_only']['company']

# Function to query Google Maps API for a hospital name
def get_google_maps_name(hospital_name, api_key):
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={hospital_name}&inputtype=textquery&fields=name&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'candidates' in data and len(data['candidates']) > 0:
            return data['candidates'][0]['name']
    return None

# Google Maps API Key
api_key = 'AIzaSyBnjZCIO5ePkHiurh-3N1aaJ__xI82uHks'

# Create a new dataframe to store the results
results = pd.DataFrame(columns=['Original Name', 'Google Maps Name'])

# %%
# Loop through the list of hospitals and get the Google Maps name for each
for hospital_name in hospitals_for_gm:
    google_maps_name = get_google_maps_name(hospital_name, api_key)
    results = results.append({'Original Name': hospital_name, 'Google Maps Name': google_maps_name}, ignore_index=True)

# Save the results to a CSV file
results.to_csv('Output_data/unmatched_hospital_google_maps_names.csv', index=False)
'''

# %% Step 3
# load results from google map search and merge with unmatched hospitals
results = pd.read_csv('Output_data/unmatched_hospital_google_maps_names.csv')
results = results.rename(columns={'Original Name':'company'})
results = results.drop_duplicates()
hospital_list2 = pd.merge(hospital_list2, results, on='company', how='left',indicator=True)
print(hospital_list2['_merge'].value_counts())
# convert to lower case
hospital_list2['Google Maps Name'] = hospital_list2['Google Maps Name'].str.lower()
# clean the google map names
hospital_list2['Google Maps Name'] = hospital_list2['Google Maps Name'].str.replace(r'[,]', '', regex=True)
hospital_list2['Google Maps Name'] = hospital_list2['Google Maps Name'].str.replace(' inc.', '', case=False)
hospital_list2['Google Maps Name'] = hospital_list2['Google Maps Name'].str.replace(' llc', '', case=False)
hospital_list2['Google Maps Name'] = hospital_list2['Google Maps Name'].str.replace(r'[\-\+&]', ' ', regex=True)
hospital_list2['Google Maps Name'] = hospital_list2['Google Maps Name'].str.replace('   ', ' ')
#
hospital_list2['company'] = hospital_list2['Google Maps Name']
#
hospital_list2 = hospital_list2.drop(columns=['_merge','Google Maps Name'])

# %%
# match google map names with Revelio companies
merged_df2 = pd.merge(company_ref, hospital_list2, on='company', how='right',indicator=True)
print(merged_df2['_merge'].value_counts())
merged_df2.to_csv('Output_data/merged_df2.csv', index=False)
# keep unmatched
hospital_list3 = merged_df2[merged_df2['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])
hospital_list3['company'] = hospital_list3['company'].str.replace('center', 'cent')
hospital_list3['company'] = hospital_list3['company'].fillna(hospital_list3['Hospital Name'].str.lower())

# %% Step 4
# find those in filtered company_ref that contain unmatched hospitals
for hospital in hospital_list3['company']:
    company_ref_filtered.loc[
        company_ref_filtered['company'].str.contains(hospital), 'company'
    ] = hospital    
    
# %%
merged_df3 = pd.merge(company_ref_filtered, hospital_list3, on='company', how='right',indicator=True)
print(merged_df3['_merge'].value_counts())
merged_df3.to_csv('Output_data/merged_df3.csv', index=False)
# keep unmatched
hospital_list4 = merged_df3[merged_df3['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])

# %% Step 5
# function to apply fuzzy matching
def fuzzy_match(row, choices, scorer, cutoff=80):
    match = process.extractOne(row, choices, scorer=scorer, score_cutoff=cutoff)
    if match:
        return match[0]  # Return the best match
    else:
        return None

# apply fuzzy matching for company_ref against hospital_list
company_ref_choices = company_ref_filtered['company'].tolist()
hospital_list4['fuzzy_match'] = hospital_list4['company'].apply(lambda x: fuzzy_match(x, company_ref_choices, fuzz.token_sort_ratio))
hospital_list4.to_csv('Output_data/hospital_list4.csv', index=False)
del company_ref_choices

# %%
# load manually corrected fuzzy matches
hospital_list4_corrected = pd.read_csv('Output_data/hospital_list4_corrected.csv')
hospital_list4_corrected['company'] = hospital_list4_corrected['fuzzy_match'].where(hospital_list4_corrected['fuzzy_match'].notna(), hospital_list4_corrected['company'])
hospital_list4_corrected = hospital_list4_corrected.drop('fuzzy_match', axis=1)

# %%
# conduct fuzzy matches
merged_df4 = pd.merge(company_ref_filtered, hospital_list4_corrected, on='company', how='right',indicator=True)
print(merged_df4['_merge'].value_counts())
merged_df4.to_csv('Output_data/merged_df4.csv', index=False)
# keep unmatched
hospital_list5 = merged_df4[merged_df4['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])

# %% Step 6
# fuzzy match for the complements of filtered company_ref
# filtering condition
condition = company_ref['company'].str.contains('hospital|medical|cardiovascular|health|psychiatric|rehabilitation|surgery|treatment|clinic|medicine|recovery|disease|surgical|care|doctor|medcenter|cancer', case=False, regex=True)

# complementary set (companies that do not match the condition)
company_ref_complementary = company_ref[~condition]
del condition

# %%
# apply fuzzy matching for company_ref against hospital_list
company_ref_choices = company_ref_complementary['company'].tolist()
del company_ref_complementary
# extract unique words from hospital_list5['company']
hospital_words = set(word for company in hospital_list5['company'] for word in company.split() if len(word) > 4)
# filter company_ref_choices based on hospital words
company_ref_choices = [company for company in company_ref_choices if any(word in company for word in hospital_words)]
# apply fuzzy matching with filtered choices
hospital_list5['fuzzy_match'] = hospital_list5['company'].apply(lambda x: fuzzy_match(x, company_ref_choices, fuzz.token_sort_ratio, 85))
hospital_list5.to_csv('Output_data/hospital_list5.csv', index=False)

# %%
# load manually corrected fuzzy matches
hospital_list5_corrected = pd.read_csv('Output_data/hospital_list5_corrected.csv')
hospital_list5_corrected['company'] = hospital_list5_corrected['fuzzy_match'].where(hospital_list5_corrected['fuzzy_match'].notna(), hospital_list5_corrected['company'])
hospital_list5_corrected = hospital_list5_corrected.drop('fuzzy_match', axis=1)

# %%
# conduct fuzzy matches
merged_df5 = pd.merge(company_ref, hospital_list5_corrected, on='company', how='right',indicator=True)
print(merged_df5['_merge'].value_counts())
merged_df5.to_csv('Output_data/merged_df5.csv', index=False)


# %%
# put all merged outcomes together
hospital_list_final = pd.merge(merged_df, hospital_list[['Hospital Name','City','Sate']], on=['Hospital Name','City','Sate'], how='right')
for i in range(1,6):
    hospital_list_final = pd.merge(eval(f'merged_df{i}'), hospital_list_final, on=['Hospital Name','City','Sate'], how='right', suffixes=('_x', '_y'))
    hospital_list_final.loc[hospital_list_final['rcid_x'].notna(), 'rcid_y'] = hospital_list_final['rcid_x']
    hospital_list_final.loc[hospital_list_final['company_x'].notna(), 'company_y'] = hospital_list_final['company_x']
    hospital_list_final.loc[hospital_list_final['file_x'].notna(), 'file_y'] = hospital_list_final['file_x']
    hospital_list_final.loc[hospital_list_final['_merge_x'].notna(), '_merge_y'] = hospital_list_final['_merge_x']
    hospital_list_final = hospital_list_final.drop(columns=['rcid_x','company_x','file_x','_merge_x'])
    hospital_list_final = hospital_list_final.rename(columns={'rcid_y': 'rcid','company_y': 'company','file_y': 'file','_merge_y': '_merge'})
    
# merge with names in company_ref
hospital_list_final = pd.merge(company_ref, hospital_list_final, on=['rcid','file'], how='right')
hospital_list_final.loc[hospital_list_final['company_x'].notna(), 'company_y'] = hospital_list_final['company_x']
hospital_list_final = hospital_list_final.drop(columns=['company_x','_merge'])
hospital_list_final = hospital_list_final.rename(columns={'company_y': 'company'})
hospital_list_final.to_csv('Output_data/hospital_list_final.csv', index=False)
 
# %%
# manually corrected errors in step 2-4 because of common hospital names
hospital_list_final_corrected = pd.read_csv('Output_data/hospital_list_final_corrected.csv')
hospital_list_final_corrected['company_orig'] = hospital_list_final_corrected['Hospital Name'].str.lower()
# clean the names
hospital_list_final_corrected['company_orig'] = hospital_list_final_corrected['company_orig'].str.replace(r'[,]', '', regex=True)
hospital_list_final_corrected['company_orig'] = hospital_list_final_corrected['company_orig'].str.replace(' inc.', '', case=False)
hospital_list_final_corrected['company_orig'] = hospital_list_final_corrected['company_orig'].str.replace(' llc', '', case=False)
hospital_list_final_corrected['company_orig'] = hospital_list_final_corrected['company_orig'].str.replace(r'[\-\+&]', ' ', regex=True)
hospital_list_final_corrected['company_orig'] = hospital_list_final_corrected['company_orig'].str.replace('   ', ' ')
#
results = pd.read_csv('Output_data/unmatched_hospital_google_maps_names.csv')
results = results.rename(columns={'Original Name':'company_orig'})
results = results.drop_duplicates()
hospital_list_final_corrected = pd.merge(hospital_list_final_corrected, results, on='company_orig', how='left')
# generate fuzzy ratios 
# function to apply fuzzy matching, handling NaN values
def safe_fuzz_ratio(val1, val2):
    if pd.isna(val1) or pd.isna(val2):
        return np.nan  # return NaN if either value is missing
    return fuzz.ratio(val1, val2)
#
hospital_list_final_corrected['fuzz_ratio'] = hospital_list_final_corrected.apply(lambda row: safe_fuzz_ratio(row['company'], row['company_orig']), axis=1)
hospital_list_final_corrected['fuzz_ratio_gm'] = hospital_list_final_corrected.apply(lambda row: safe_fuzz_ratio(row['company'], row['Google Maps Name']), axis=1)
hospital_list_final_corrected.to_csv('Output_data/hospital_list_final_corrected1.csv', index=False)

# %% Clean the list
# manually corrected matches with low fuzzy ratios
hospital_list_final_corrected1 = pd.read_csv('Output_data/hospital_list_final_corrected1.csv')
# assign a unique ID based on hospital_name and city
hospital_list_final_corrected1['id'] = hospital_list_final_corrected1.groupby(['Hospital Name', 'City', 'Sate']).ngroup()

# step 1: create a mask for hospitals where at least one rcid is non-missing
non_missing_rcid = hospital_list_final_corrected1.groupby('id')['rcid'].transform(lambda x: x.notna().any())

# step 2: keep all rows where rcid is not missing or where the hospital has no non-missing values
hospital_list_final_corrected1 = hospital_list_final_corrected1[(hospital_list_final_corrected1['rcid'].notna()) | (~non_missing_rcid)]

# step 3: for hospitals where all rcid values are missing, keep only the first row
hospital_list_final_corrected1 = hospital_list_final_corrected1[hospital_list_final_corrected1['rcid'].notna() | (~non_missing_rcid & hospital_list_final_corrected1.duplicated(subset='id', keep='first') == False)]

hospital_list_final_corrected1 = hospital_list_final_corrected1.drop(columns=['id', 'n'])
hospital_list_final_corrected1.to_csv('Output_data/hospital_list_final_corrected2.csv', index=False)

# %% For quality check
# Assign a unique ID based on hospital_name and city
random = hospital_list_final_corrected1
random['id'] = random.groupby(['Hospital Name', 'City', 'Sate']).ngroup()
# randomly choose 100 matched
# Step 1: Filter for non-missing rcid
non_missing_rcid = random[random['rcid'].notna()]

# Step 2: Randomly select 100 unique hospital names
sampled_matched_hospitals = non_missing_rcid['id'].drop_duplicates().sample(n=100, random_state=42)

# Step 3: Retrieve all rows corresponding to the sampled hospitals
sampled_matched_hospitals = non_missing_rcid[non_missing_rcid['id'].isin(sampled_matched_hospitals)]
sampled_matched_hospitals = sampled_matched_hospitals.drop(columns=['id'])

# randomly choose 100 unmatched
# Step 1: Filter for missing rcid
missing_rcid = random[random['rcid'].isna()]

# Step 2: Randomly select 50 unique hospital names
sampled_unmatched_hospitals = missing_rcid['id'].sample(n=50, random_state=42)

# Step 3: Retrieve all rows corresponding to the sampled hospitals
sampled_unmatched_hospitals = missing_rcid[missing_rcid['id'].isin(sampled_unmatched_hospitals)]
sampled_unmatched_hospitals = sampled_unmatched_hospitals.drop(columns=['id'])

# save
with pd.ExcelWriter('Output_data/hospital_list_final_corrected2.xlsx') as writer:  # doctest: +SKIP
    hospital_list_final_corrected1.to_excel(writer, sheet_name='All hospitals', index=False)
    sampled_matched_hospitals.to_excel(writer, sheet_name='100 matched', index=False)
    sampled_unmatched_hospitals.to_excel(writer, sheet_name='50 unmatched', index=False)


# %% Redo the fuzzy match for those corrected
hospital_list6 = hospital_list_final_corrected1[hospital_list_final_corrected1['rcid'].isna()][['Hospital Name', 'City', 'Sate', 'company']]
hospital_list6['company_'] = hospital_list6['company']
hospital_list6['company'] = hospital_list6['Hospital Name'].str.lower()
# clean the names
hospital_list6['company'] = hospital_list6['company'].str.replace(r'[,]', '', regex=True)
hospital_list6['company'] = hospital_list6['company'].str.replace(' inc.', '', case=False)
hospital_list6['company'] = hospital_list6['company'].str.replace(' llc', '', case=False)
hospital_list6['company'] = hospital_list6['company'].str.replace(r'[\-\+&]', ' ', regex=True)
hospital_list6['company'] = hospital_list6['company'].str.replace('   ', ' ')
hospital_list6 = hospital_list6[hospital_list6['company'] != hospital_list6['company_']]
hospital_list6 = hospital_list6.drop(columns=['company_'])
# apply fuzzy matching for company_ref against hospital_list
company_ref_choices = company_ref_filtered['company'].tolist()
hospital_list6['fuzzy_match'] = hospital_list6['company'].apply(lambda x: fuzzy_match(x, company_ref_choices, fuzz.token_sort_ratio))
hospital_list6.to_csv('Output_data/hospital_list6.csv', index=False)
del company_ref_choices

# %%
# load manually corrected fuzzy matches
hospital_list6_corrected = pd.read_excel('Output_data/hospital_list6_corrected.xlsx')
hospital_list6_corrected['company'] = hospital_list6_corrected['fuzzy_match'].where(hospital_list6_corrected['fuzzy_match'].notna(), hospital_list6_corrected['company'])
hospital_list6_corrected = hospital_list6_corrected.drop('fuzzy_match', axis=1)

# %%
# conduct fuzzy matches
merged_df6 = pd.merge(company_ref_filtered, hospital_list6_corrected, on='company', how='right',indicator=True)
print(merged_df6['_merge'].value_counts())
merged_df6.to_csv('Output_data/merged_df6.csv', index=False)
# keep unmatched
hospital_list7 = merged_df6[merged_df6['_merge'] == 'right_only'].drop(columns=['rcid', 'file', '_merge'])

# %%
# fuzzy match for the complements of filtered company_ref
# filtering condition
condition = company_ref['company'].str.contains('hospital|medical|cardiovascular|health|psychiatric|rehabilitation|surgery|treatment|clinic|medicine|recovery|disease|surgical|care|doctor|medcenter|cancer', case=False, regex=True)

# complementary set (companies that do not match the condition)
company_ref_complementary = company_ref[~condition]
del condition

# %%
# apply fuzzy matching for company_ref against hospital_list
company_ref_choices = company_ref_complementary['company'].tolist()
del company_ref_complementary
# extract unique words from hospital_list7['company']
hospital_words = set(word for company in hospital_list7['company'] for word in company.split() if len(word) > 4)
# filter company_ref_choices based on hospital words
company_ref_choices = [company for company in company_ref_choices if any(word in company for word in hospital_words)]
# apply fuzzy matching with filtered choices
hospital_list7['fuzzy_match'] = hospital_list7['company'].apply(lambda x: fuzzy_match(x, company_ref_choices, fuzz.token_sort_ratio, 85))
hospital_list7.to_csv('Output_data/hospital_list7.csv', index=False)
del company_ref_choices

# %%
# load manually corrected fuzzy matches
hospital_list7_corrected = pd.read_csv('Output_data/hospital_list7_corrected.csv')
hospital_list7_corrected['company'] = hospital_list7_corrected['fuzzy_match'].where(hospital_list7_corrected['fuzzy_match'].notna(), hospital_list7_corrected['company'])
hospital_list7_corrected = hospital_list7_corrected.drop('fuzzy_match', axis=1)

# %%
# conduct fuzzy matches
merged_df7 = pd.merge(company_ref, hospital_list7_corrected, on='company', how='right',indicator=True)
print(merged_df7['_merge'].value_counts())
merged_df7.to_csv('Output_data/merged_df7.csv', index=False)

# %%
# put all together
hospital_list_final_corrected2 = hospital_list_final_corrected1
hospital_list_final_corrected2['_merge'] = 0
for i in range(6,8):
    hospital_list_final_corrected2 = pd.merge(eval(f'merged_df{i}'), hospital_list_final_corrected2, on=['Hospital Name','City','Sate'], how='right', suffixes=('_x', '_y'))
    hospital_list_final_corrected2.loc[hospital_list_final_corrected2['rcid_x'].notna(), 'rcid_y'] = hospital_list_final_corrected2['rcid_x']
    hospital_list_final_corrected2.loc[hospital_list_final_corrected2['company_x'].notna(), 'company_y'] = hospital_list_final_corrected2['company_x']
    hospital_list_final_corrected2.loc[hospital_list_final_corrected2['file_x'].notna(), 'file_y'] = hospital_list_final_corrected2['file_x']
    hospital_list_final_corrected2.loc[hospital_list_final_corrected2['_merge_x'].notna(), '_merge_y'] = hospital_list_final_corrected2['_merge_x']
    hospital_list_final_corrected2 = hospital_list_final_corrected2.drop(columns=['rcid_x','company_x','file_x','_merge_x'])
    hospital_list_final_corrected2 = hospital_list_final_corrected2.rename(columns={'rcid_y': 'rcid','company_y': 'company','file_y': 'file','_merge_y': '_merge'})
    
hospital_list_final_corrected2 = pd.merge(company_ref, hospital_list_final_corrected2, on=['rcid','file'], how='right')
hospital_list_final_corrected2.loc[hospital_list_final_corrected2['company_x'].notna(), 'company_y'] = hospital_list_final_corrected2['company_x']
hospital_list_final_corrected2 = hospital_list_final_corrected2.drop(columns=['company_x','_merge'])
hospital_list_final_corrected2 = hospital_list_final_corrected2.rename(columns={'company_y': 'company'})

hospital_list_final_corrected2.to_excel('Output_data/hospital_list_final_corrected3.xlsx', index=False)
hospital_list_final_corrected3 = hospital_list_final_corrected2[['rcid','Hospital Name','City','Sate','company']]

# %% Match completed
# merge with all company information
# import company ref data
# define the folder path containing the files
folder_path = 'F:/academic_company_ref/*.parquet'
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
    df = df.drop(columns=['company'])
    df_list.append(df)
    i += 1
# concatenate all DataFrames in the list
company_ref = pd.concat(df_list, ignore_index=True)
del parquet, df, df_list

# %%
hospital_list_final_corrected3 = pd.merge(company_ref, hospital_list_final_corrected3, on=['rcid'], how='right',indicator=True)
print(hospital_list_final_corrected3['_merge'].value_counts())
hospital_list_final_corrected3 = hospital_list_final_corrected3.drop(columns=['_merge'])
hospital_list_final_corrected3.to_excel('Output_data/hospital_list_company_ref.xlsx', index=False)
del company_ref, company_ref_filtered

# # %% To find potentially problematic matches in step 2 and 4
# hs = hospital_list[['company']]
# hs['ind'] = 0
# hs['c'] = ""
# for hospital in hospital_list['company']:
#     hs.loc[
#         (hs['company'].str.contains(hospital)) & (hs['company'] != hospital), 'ind'
#     ] += 1  
#     hs.loc[
#         (hs['company'].str.contains(hospital)) & (hs['company'] != hospital), 'c'
#     ] = hospital
#     target_string = "calais community hospital"
# hs['ind_'] = hs['company'].apply(lambda x: 1 if target_string.lower().find(x.lower()) != -1 else 0)

# # Get the current time
# current_time = datetime.datetime.now()

# # Print the current time
# print("Current time:", current_time)

