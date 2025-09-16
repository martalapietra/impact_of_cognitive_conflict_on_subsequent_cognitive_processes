# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:20:57 2025

@author: Marta La Pietra
Project: The "sweet spot" of Cognitive Conflict (PID2020-114717RA-I00)
Description: This script analyzes the data from the Experiment 3 (Stroop task intermixed with a semantic categorisation task + an impromptu implicit memory task) in the Registered Report titled "Exploring the impact of cognitive conflict on subsequent cognitive processes". 
It calculates the mean reaction times (RTs) and accuracies for for the Stroop task, the accuracy for the semantic categorisation and the implicit memory task. 
The script also performs statistical tests to compare the effects of different Stroop trial types on accuracy at the implicit memory task.
"""

# LOAD PACKAGES #
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Directory where you want to store the data files
datapath = 'GitHub/' # select the proper datapath
foldername = "data"

filename = "exp3_all_trials.csv"
file_path = os.path.join(datapath, foldername, filename)
full_data = pd.read_csv(file_path, header=None)

# Insert column names 
full_data.columns=['N', 'Word' ,'Color' ,'Congruency' ,'Response' ,'Accuracy', 'RT', 'Condition', 'RespCategory', 'RT2', 'AccuracyCategory', 'ImageFilename', 'P']
# Move the last column to the first position
last_col = full_data.columns[-1]
full_data = full_data[[last_col] + list(full_data.columns[:-1])]

# Map the Stroop trial type
congruency_mapping = {1: 'Congruent', 0: 'Incongruent', 2: 'Neutral'}
full_data['Congruence'] = full_data['Congruency'].map(congruency_mapping)

# Sort by Participant (P) and Trials (N)
df = full_data.sort_values(by=["P", "N"])

# Define the function to assign block numbers
def assign_blocks(df, trials_per_block=36):
    # Initialize the Block column
    df['Block'] = 0
    # Loop over each participant's group
    for participant, group in df.groupby('P'):
        # Calculate the block number for each trial
        blocks = (group['N'] - 1) // trials_per_block + 1
        # Assign the block numbers to the main dataframe using the original index
        df.loc[group.index, 'Block'] = blocks    
    return df
# Apply the function
df = assign_blocks(df, trials_per_block=36)

# Identify the current order of columns
columns = list(df.columns)
# Rebuild the column list by keeping the first two, adding 'Block', then the rest
columns = columns[:2] + ['Block'] + columns[2:-1]
# Reassign the columns to the dataframe
df = df[columns]

# Identify the first trial of each block
df['Excluded'] = (df['N'] == 1).astype(int)
first_trials = [1 + 36 * (i - 1) for i in range(1, 11)]
# Assign 1 to 'excluded' column for these trials
df['Excluded'] = df['N'].apply(lambda x: 1 if x in first_trials else 0)

# Calculate how many Stroop trials are excluded
excluded_stroop = df.loc[(df['Accuracy'] == 0) | (df['Excluded'] != 0) | (df['RT'] >= 1.500)]
# Select only the secondary task
categorization = df[df['Condition'] != 0].copy()
# Calculate how many secondary trials are excluded following the Stroop
categorization_excluded = excluded_stroop[excluded_stroop['Condition'] != 0].copy()

# Save first trials of each block for the Stroop task
excluded_first = df[df['Excluded'] != 0].copy()
# And exclude them from the dataframe
df = df[df['Excluded'] == 0].copy()    

# Save too slow RTs to the Stroop task
excluded_tooslow = df[df['RT'] > 1.5].copy()
# And exclude them 
df = df[df['RT'] < 1.5].copy()    

# Save too fast RTs to the Stroop task
excluded_toofast = df[df['RT']<= 0.05].copy()
# And exclude them 
df = df[df['RT'] >= 0.05].copy() 

# Overall accuracy at the Stroop task
overall_acc = df['Accuracy'].mean() * 100
overall_acc_std = df['Accuracy'].std() * 100
print("\nOverall Accuracy at the Stroop task in Experiment 3:", overall_acc, "±", overall_acc_std)

# Calculate accuracy for each trial type and participant
def calculate_accuracy(df):
    # Calculate accuracy based on Conflict_Level and Congruency
    accuracy_results = []
    # Group data by Conflict_Level and Congruency
    grouped = df.groupby(['P', 'Congruence'])
    for (participant, congruency), group_data in grouped:
        # Calculate accuracy for each group
        total_trials = len(group_data)
        correct_trials = group_data['Accuracy'].sum()
        accuracy = (correct_trials / total_trials) * 100  # Convert to percentage
        std_accuracy = group_data['Accuracy'].std()*100
        # Store accuracy result
        accuracy_results.append({'Participant': participant,
                                 'Congruence': congruency,
                                 'Accuracy': accuracy,
                                 'Std': std_accuracy})
    # Create a DataFrame from accuracy results
    accuracy_df = pd.DataFrame(accuracy_results)
    return accuracy_df

# Call the function to calculate accuracy
accuracy_df = calculate_accuracy(df)

# Remove erroneous answers to the Stroop task
clean_df = df[df['Accuracy'] != 0].copy()

# RTs on correct trials
def calculate_rt_stats(df):
    # Create a list to store the results
    results = []
    # Group the DataFrame by Participant Number
    grouped = df.groupby('P')
    for participant, data in grouped:
        # Calculate mean RTs for each Congruency condition
        overallmeanRTs = data['RT'].mean()
        mean_rt_congruent = data[data['Congruency'] == 1]['RT'].mean()
        mean_rt_incongruent = data[data['Congruency'] == 0]['RT'].mean()
        mean_rt_neutral = data[data['Congruency'] == 2]['RT'].mean()
        # Calculate the conflict effect
        conflict_effect = mean_rt_incongruent - mean_rt_congruent
        # Append the results to the list
        results.append({
            'Participant': participant,
            'Overall_Mean_RTs': overallmeanRTs,
            'Mean_RTs_Congruent': mean_rt_congruent,
            'Mean_RTs_Incongruent': mean_rt_incongruent,
            'Mean_RTs_Neutral': mean_rt_neutral,
            'Conflict_Effect': conflict_effect
        })
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Call the function to calculate RTs
results_df = calculate_rt_stats(clean_df)

# Calculate mean RTs for each trial type and participant
overall_rts = results_df['Overall_Mean_RTs']
congruent_rts = results_df['Mean_RTs_Congruent']
incongruent_rts = results_df['Mean_RTs_Incongruent']
neutral_rts = results_df['Mean_RTs_Neutral']
stroop_effect = results_df['Conflict_Effect'] 
overall_rts_mean = overall_rts.mean() *1000
overall_rts_sd = overall_rts.std()*1000
congruent_rts_mean = congruent_rts.mean() *1000
congruent_rts_sd = congruent_rts.std()*1000
incongruent_rts_mean = incongruent_rts.mean()*1000
incongruent_rts_sd = incongruent_rts.std()*1000
neutral_rts_mean = neutral_rts.mean()*1000
neutral_rts_sd = neutral_rts.std()*1000
stroop_effect_mean = stroop_effect.mean()*1000
stroop_effect_sd = stroop_effect.std()*1000
print(f"\nOverall Mean RTs: {overall_rts_mean:.2f} ± {overall_rts_sd:.2f} ms")
print(f"Overall Stroop Effect: {stroop_effect_mean:.2f} ± {stroop_effect_sd:.2f} ms")
print(f"Overall Mean RTs for CONGRUENT Trials: {congruent_rts_mean:.2f} ± {congruent_rts_sd:.2f} ms")
print(f"Overall Mean RTs for INCONGRUENT Trials: {incongruent_rts_mean:.2f} ± {incongruent_rts_sd:.2f} ms")
print(f"Overall Mean RTs for NEUTRAL Trials: {neutral_rts_mean:.2f} ± {neutral_rts_sd:.2f} ms\n")

###################### ANOVA STROOP
selected_columns = results_df[['Mean_RTs_Congruent', 'Mean_RTs_Incongruent', 'Mean_RTs_Neutral']]

long_data = selected_columns.melt(var_name='TrialType', value_name='RTs')

# Perform one-way ANOVA with Stroop effect x 3 Experiments
model = ols('RTs ~ C(TrialType)', data=long_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Calculate Eta-Squared
SSB = anova_table['sum_sq']['C(TrialType)']  # Sum of Squares Between Groups
SST = anova_table['sum_sq'].sum()                 # Total Sum of Squares (Between + Within)
eta_squared = SSB / SST

print(f"Eta-Squared (η²): {eta_squared}\n")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's HSD test for the Experiment (Task) factor
tukey = pairwise_tukeyhsd(endog=long_data['RTs'], groups=long_data['TrialType'], alpha=0.05)
print(tukey)

# Perform one-tailed paired t-tests (p < 0.05) to compare the RTs for each trial type
t_cong_incong = ttest_rel(incongruent_rts, congruent_rts)
t_incong_neutral = ttest_rel(incongruent_rts, neutral_rts)
t_cong_neutral = ttest_rel(congruent_rts, neutral_rts)

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_cong_incong = cohen_d(incongruent_rts, congruent_rts)
d_incong_neutral = cohen_d(incongruent_rts, neutral_rts)
d_cong_neutral = cohen_d(congruent_rts, neutral_rts)

# Print t-test and Cohen's d results
print("CONFLICT EFFECT: One-tailed paired t-test (p < 0.05)")
print("Incongruent vs. Congruent:")
print("t-statistic:", t_cong_incong.statistic)
print("p-value:", t_cong_incong.pvalue)
print("Cohen's d:", d_cong_incong)

print("\nIncongruent vs. Neutral:")
print("t-statistic:", t_incong_neutral.statistic)
print("p-value:", t_incong_neutral.pvalue)
print("Cohen's d:", d_incong_neutral)

print("\nCongruent vs. Neutral:")
print("t-statistic:", t_cong_neutral.statistic)
print("p-value:", t_cong_neutral.pvalue)
print("Cohen's d:", d_cong_neutral,"\n")

# Store the mean RTs at the Stroop task for each participants and trial type
stroop_meanRTs = clean_df.groupby(["P","Congruence"])
stroop_meanRTs = stroop_meanRTs['RT'].mean()
stroop_meanRTs = stroop_meanRTs.reset_index()

#-------------------------------------------------------
# SECONDARY TASK
# Save and the trials that were responded too fast
excluded_category = categorization[categorization['RT2'] <= 0.05].copy() 
categorization = categorization[categorization['RT2'] >= 0.05].copy() 

# Check accuracy at the semantic categorisation
def calculate_accuracy(df):
    # Calculate accuracy based on Conflict_Level and Congruency
    accuracy_results = []

    # Group data by Conflict_Level and Congruency
    grouped = df.groupby(['P', 'Congruence'])

    for (participant, congruency), group_data in grouped:
        # Calculate accuracy for each group
        total_trials = len(group_data)
        correct_trials = group_data['AccuracyCategory'].sum()
        accuracy = (correct_trials / total_trials) * 100  # Convert to percentage
        std_accuracy = group_data['AccuracyCategory'].std()*100
        RTs = group_data['RT2'].mean()
        RTs_std = group_data['RT2'].std()

        # Store accuracy result
        accuracy_results.append({'Participant': participant,
                                 'Congruence': congruency,
                                 'Accuracy_Categorisation': accuracy,
                                 'Accuracy std': std_accuracy,
                                 'MeanRT_Categorisation': RTs,
                                 'RTs std': RTs_std})

    # Create a DataFrame from accuracy results
    accuracy_df = pd.DataFrame(accuracy_results)

    return accuracy_df

# Call the function to calculate accuracy
accuracy_categorisation = calculate_accuracy(categorization)

def calculate_accuracy_overall(df):
    # Calculate accuracy based on Conflict_Level and Congruency
    accuracy_results = []
    # Group data by Conflict_Level and Congruency
    grouped = df.groupby('P')
    for (participant), group_data in grouped:
        # Calculate accuracy for each group
        total_trials = len(group_data)
        correct_trials = group_data['AccuracyCategory'].sum()
        accuracy = (correct_trials / total_trials) * 100  # Convert to percentage
        std_accuracy = group_data['AccuracyCategory'].std()*100
        RTs = group_data['RT2'].mean()
        RTs_std = group_data['RT2'].std()
        # Store accuracy result
        accuracy_results.append({'Participant': participant,
                                 'Accuracy_Categorisation': accuracy,
                                 'Accuracy std': std_accuracy,
                                 'MeanRT_Categorisation': RTs,
                                 'RTs std': RTs_std})
    # Create a DataFrame from accuracy results
    accuracy_df = pd.DataFrame(accuracy_results)
    return accuracy_df
     
# Call the function to calculate accuracy
accuracy_categorisation_overall = calculate_accuracy_overall(categorization)

combined_accuracy_categorisation = pd.concat([accuracy_categorisation_overall, accuracy_categorisation], ignore_index=True)
combined_accuracy_categorisation['Congruence'].fillna('Overall', inplace = True)
# Define the custom order for the 'Congruency' column
congruency_order = ['Congruent', 'Incongruent', 'Neutral', 'Overall']

# Convert 'Congruency' to categorical data type with the custom order
combined_accuracy_categorisation['Congruence'] = pd.Categorical(combined_accuracy_categorisation['Congruence'], categories=congruency_order, ordered=True)
# Sort the combined accuracy dataframe by 'Participant' and 'Block' columns
combined_accuracy_categorisation.sort_values(by=['Participant','Congruence'], inplace=True)

#--------------------------------- Load the dataframe from the Implicit Memory Task!
datapath_memory = 'GitHub/'
foldername_memory = "data"

filename_memory  = "exp3_all_trials_secondary.csv"
file_path_memory = os.path.join(datapath_memory, foldername_memory, filename_memory)
full_data_memory = pd.read_csv(file_path_memory, header=None)
# Insert column names 
full_data_memory.columns=['P', 'Image1' ,'Image2' ,'Condition', 'ScreenSide', 'Response', 'Accuracy', 'RT']

# Select only the Images presented, the accuracy and the RTs
image_memory_df = full_data_memory[['P', 'Image1' ,'Image2','Accuracy','RT']]
# Select the relevant columns from the categorisation dataset
image_df = categorization[['P', 'N', 'Block','AccuracyCategory', 'ImageFilename','Congruence','RT2']]

#Create a single dataframe with all the info
merged_df = pd.merge(image_df, image_memory_df, on='P')

# Check if 'Image1' matches 'ImageFilename'
merged_df['Image1_Match'] = merged_df['Image1'] == merged_df['ImageFilename']

# Check if 'Image2' matches 'ImageFilename'
merged_df['Image2_Match'] = merged_df['Image2'] == merged_df['ImageFilename']

matched_rows_image1 = merged_df[(merged_df['Image1_Match'] == True)].reset_index()
matched_rows_image2 = merged_df[(merged_df['Image2_Match'] == True)].reset_index()

matched_rows_image1 = matched_rows_image1.drop(columns=['Image2', 'Image1_Match','Image2_Match'])
matched_rows_image2 = matched_rows_image2.drop(columns=['Image1', 'Image1_Match','Image2_Match'])

matched_rows_image1 = matched_rows_image1.rename(columns={"Image1": "Image"})
matched_rows_image2 = matched_rows_image2.rename(columns={"Image2": "Image"})

matched_rows_image1['ImageType'] = 'Image1'
matched_rows_image2['ImageType'] = 'Image2'
combined_df = pd.concat([matched_rows_image1, matched_rows_image2], ignore_index=True)
combined_df = combined_df.drop(columns=['index'])

excluded_memory = combined_df[combined_df['RT'] <= 0.05].copy() 
combined_df = combined_df[combined_df['RT'] > 0.05].copy() 

# Accuracy at the implicit memory test
def calculate_accuracy(df):
    # Calculate accuracy based on Conflict_Level and Congruency
    accuracy_results = []
    # Group data by Conflict_Level and Congruency
    grouped = df.groupby(['P', 'Congruence'])

    for (participant, congruency), group_data in grouped:
        # Calculate accuracy for each group
        total_trials = len(group_data)
        correct_trials = group_data['Accuracy'].sum()
        accuracy = (correct_trials / total_trials) * 100  # Convert to percentage
        std_accuracy = group_data['Accuracy'].std()*100
        RTs = group_data['RT'].mean()
        RTs_std = group_data['RT'].std()

        # Store accuracy result
        accuracy_results.append({'P': participant,
                                 'Congruence': congruency,
                                 'Accuracy': accuracy,
                                 'Accuracy std': std_accuracy,
                                 'MeanRT': RTs,
                                 'RTs std': RTs_std})

    # Create a DataFrame from accuracy results
    accuracy_df = pd.DataFrame(accuracy_results)
    return accuracy_df
# Call the function to calculate accuracy
accuracy_memory = calculate_accuracy(combined_df)

def calculate_accuracy_overall(df):
    # Calculate accuracy based on Conflict_Level and Congruency
    accuracy_results = []
    # Group data by Conflict_Level and Congruency
    grouped = df.groupby('P')
    for (participant), group_data in grouped:
        # Calculate accuracy for each group
        total_trials = len(group_data)
        correct_trials = group_data['Accuracy'].sum()
        accuracy = (correct_trials / total_trials) * 100  # Convert to percentage
        std_accuracy = group_data['Accuracy'].std()*100
        RTs = group_data['RT'].mean()
        RTs_std = group_data['RT'].std()
        # Store accuracy result
        accuracy_results.append({'P': participant,
                                 'Accuracy': accuracy,
                                 'Accuracy std': std_accuracy,
                                 'MeanRT': RTs,
                                 'RTs std': RTs_std})
    # Create a DataFrame from accuracy results
    accuracy_df = pd.DataFrame(accuracy_results)
    return accuracy_df
      
# Call the function to calculate accuracy
accuracy_memory_overall = calculate_accuracy_overall(combined_df)

combined_accuracy_memory = pd.concat([accuracy_memory_overall, accuracy_memory], ignore_index=True)
combined_accuracy_memory['Congruence'].fillna('Overall', inplace = True)
# Define the custom order for the 'Congruency' column
congruency_order = ['Congruent', 'Incongruent', 'Neutral', 'Overall']

# Convert 'Congruency' to categorical data type with the custom order
combined_accuracy_memory['Congruence'] = pd.Categorical(combined_accuracy_memory['Congruence'], categories=congruency_order, ordered=True)

# Sort the combined accuracy dataframe by 'Participant' and 'Block' columns
combined_accuracy_memory.sort_values(by=['P','Congruence'], inplace=True)

# Calculate the accuracy at the memory task for the images presented after each Stroop trial type
memory_after_incongr = combined_df[combined_df['Congruence'] == "Incongruent"].copy()
memory_after_congr = combined_df[combined_df['Congruence'] == "Congruent"].copy()
memory_after_neutral = combined_df[combined_df['Congruence'] == "Neutral"].copy()

accuracy_memory_after_incongr = combined_accuracy_memory[combined_accuracy_memory['Congruence'] == 'Incongruent']['Accuracy']
accuracy_memory_after_congr = combined_accuracy_memory[combined_accuracy_memory['Congruence'] == 'Congruent']['Accuracy']
accuracy_memory_after_neutral = combined_accuracy_memory[combined_accuracy_memory['Congruence'] == 'Neutral']['Accuracy']

mean_memory_after_congr = accuracy_memory_after_congr.mean() 
std_memory_after_congr = accuracy_memory_after_congr.std() 
print("\nAccuracy at the implicit memory task after Stroop congruent trials:", mean_memory_after_congr, "±", std_memory_after_congr)

mean_memory_after_incongr = accuracy_memory_after_incongr.mean() 
std_memory_after_incongr = accuracy_memory_after_incongr.std()
print("\nAccuracy at the implicit memory task after Stroop congruent trials:", mean_memory_after_incongr, "±", std_memory_after_incongr)

mean_memory_after_neutral = accuracy_memory_after_neutral.mean() 
std_memory_after_neutral = accuracy_memory_after_neutral.std() 
print("\nAccuracy at the implicit memory task after Stroop congruent trials:", mean_memory_after_neutral, "±", std_memory_after_neutral)

# Perform registered two-tailed paired t-test (p < 0.05) to compare the accuracy at the memory task after incongruent and congruent Stroop trials
t_cong_incong = ttest_rel(accuracy_memory_after_incongr, accuracy_memory_after_congr, alternative='two-sided')

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_cong_incong = cohen_d(accuracy_memory_after_incongr, accuracy_memory_after_congr)

# Print t-test and Cohen's d results
print("REGISTERED DATA ANALYSIS: Two-tailed paired t-test (p < 0.05)")
print("After Incongruent vs. After Congruent:")
print("t-statistic:", t_cong_incong.statistic)
print("p-value:", t_cong_incong.pvalue)
print("Cohen's d:", d_cong_incong)

# Perform registered two-tailed paired t-test (p < 0.05) to compare the RTs after incongruent and congruent Stroop trials
t_incong_neutralRT = ttest_rel(accuracy_memory_after_incongr, accuracy_memory_after_neutral, alternative='two-sided')
t_cong_neutralRT = ttest_rel(accuracy_memory_after_congr, accuracy_memory_after_neutral, alternative='two-sided')

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_incong_neutralRT = cohen_d(accuracy_memory_after_incongr, accuracy_memory_after_neutral)
d_cong_neutralRT = cohen_d(accuracy_memory_after_congr, accuracy_memory_after_neutral)

# Print t-test and Cohen's d results
print("EXPLORATORY: T-tests comparing the mean Accuracy after Stroop trial types.")
print("\nAfter Incongruent vs. After Neutral:")
print("t-statistic:", t_incong_neutralRT.statistic)
print("p-value:", t_incong_neutralRT.pvalue)
print("Cohen's d:", d_incong_neutralRT)
print("\nAfter Congruent vs. After Neutral:")
print("t-statistic:", t_cong_neutralRT.statistic)
print("p-value:", t_cong_neutralRT.pvalue)
print("Cohen's d:", d_cong_neutralRT, "\n")

###################### ANOVA
# Perform one-way ANOVA with Stroop effect x 3 Experiments
model_secondary = ols('Accuracy ~ C(Congruence)', data=accuracy_memory).fit()
anova_table_secondary = sm.stats.anova_lm(model_secondary, typ=2)
print(anova_table_secondary)
model_results = model_secondary.summary()
print(model_results)

tukey_oneway = pairwise_tukeyhsd(endog = accuracy_memory["Accuracy"], groups = accuracy_memory["Congruence"])
# Display the results
print(tukey_oneway.summary())

# Calculate Eta-Squared
SSB = anova_table_secondary['sum_sq']['C(Congruence)']  # Sum of Squares Between Groups
SST = anova_table_secondary['sum_sq'].sum()                 # Total Sum of Squares (Between + Within)
eta_squared = SSB / SST

print(f"Eta-Squared (η²): {eta_squared}\n")

# SAVE RESULTS FILE
# Store file in the output folder
resultspath = 'GitHub/data/' # select the proper datapath

# Clean datasets to perform exploratory analyses: Stroop task
excel_output_path = os.path.join(resultspath, 'exp3_stroop_clean.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    clean_df.to_excel(writer, sheet_name='exp3_stroop_clean', index=False)
print(f"Results saved to {excel_output_path}")

# Clean datasets to perform exploratory analyses: Stroop task
excel_output_path = os.path.join(resultspath, 'exp3_stroop_meanRTs.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    stroop_meanRTs.to_excel(writer, sheet_name='exp3_stroop_meanRTs', index=False)
print(f"Results saved to {excel_output_path}")

# Clean datasets to perform exploratory analyses: Stroop task
excel_output_path = os.path.join(resultspath, 'exp3_secondary.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    combined_df.to_excel(writer, sheet_name='exp3_secondary', index=False)
print(f"Results saved to {excel_output_path}")

# Clean datasets to perform exploratory analyses: Stroop task
excel_output_path = os.path.join(resultspath, 'exp3_secondary_summary.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    accuracy_memory.to_excel(writer, sheet_name='exp3_secondary_summary', index=False)
print(f"Results saved to {excel_output_path}")
