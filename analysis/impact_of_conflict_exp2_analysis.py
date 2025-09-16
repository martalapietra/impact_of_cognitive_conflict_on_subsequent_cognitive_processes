# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:22:32 2024

@author: Marta La Pietra
Project: The "sweet spot" of Cognitive Conflict (PID2020-114717RA-I00)
Description: This script analyzes the data from the Experiment 2 (Stroop task intermixed with a Go/No-Go task) in the Registered Report titled "Exploring the impact of cognitive conflict on subsequent cognitive processes". 
It calculates the mean reaction times (RTs) and accuracies for for the Stroop task and the Go/No-Go taks including GO, NO-GO, Miss, and False Alarm. 
The script also performs statistical tests to compare the effects of different Stroop trial types on RTs at the secondary task.
"""
# LOAD PACKAGES #
import os # to manage file paths
import pandas as pd # to handle dataframes
import numpy as np # to handle arrays
from scipy.stats import ttest_rel # to perform t-tests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Directory where you want to store the data files
datapath = 'GitHub/' # select the proper datapath
foldername = "data"

filename = "exp2_all_trials.csv"
file_path = os.path.join(datapath, foldername, filename)
df = pd.read_csv(file_path)

# Count occurrences of each Type for each participant
type_counts = df.groupby(['P', 'Type']).size().unstack(fill_value=0)

# Convert the result to a DataFrame (optional)
result_df = type_counts.reset_index()
result_df['TotalGo'] = result_df['GO'] + result_df['Miss']
result_df['TotalNoGo'] = result_df['NO-GO'] + result_df['False Alarm']

# Create the 'Group' column in result_df
result_df['Group'] = result_df.apply(
    lambda row: 1 if row['TotalGo'] == 240 and row['TotalNoGo'] == 120 else 
                (2 if row['TotalGo'] == 200 and row['TotalNoGo'] == 160 else 'Other'),
    axis=1
)

# Merge the result_df (with Group) into the full data frame (df)
df_with_groups = pd.merge(df, result_df[['P', 'Group']], on='P', how='left')
# Count the number of participants in each group
group_counts = df_with_groups.groupby('Group')['P'].nunique().reset_index(name='Participant Count')

# Display the result
print(group_counts)

# Clean the dataset from RTs > 1.50 s and < 0.05 s in the Stroop task
clean_all = df_with_groups[df_with_groups['RT'] >= 0.05].copy()
clean_all = clean_all[clean_all['RT']< 1.5].copy()
# Remove first trials of each block 
clean_all = clean_all[clean_all['Excluded'] == 0]

# Remove erroneous answers to the Stroop task
clean_all = clean_all[clean_all['Accuracy'] != 0].copy()

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
                                  'Congruency': congruency,
                                  'Accuracy': accuracy,
                                  'Std': std_accuracy})

    # Create a DataFrame from accuracy results
    accuracy_df = pd.DataFrame(accuracy_results)

    return accuracy_df

# Call the function to calculate accuracy
accuracy_df = calculate_accuracy(df)

# Calculate the mean RTs for each trial type and participant
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
results_df = calculate_rt_stats(clean_all)

# Calculate the mean RTs for each trial type and participant
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

###################### ANOVA
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

# Perform one-tailed paired t-tests (p < 0.05) to compare the RTs for each trial type
t_cong_incong = ttest_rel(incongruent_rts, congruent_rts, alternative='two-sided')
t_incong_neutral = ttest_rel(incongruent_rts, neutral_rts, alternative='two-sided')
t_cong_neutral = ttest_rel(congruent_rts, neutral_rts, alternative='two-sided')

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
stroop_meanRTs = clean_all.groupby(["P","Congruence"])
stroop_meanRTs = stroop_meanRTs['RT'].mean()
stroop_meanRTs = stroop_meanRTs.reset_index()

# -------------------------------------------------------
# SECONDARY TASK
# Store excluded RTs to the secondary task in another dataset
excluded_go_secondary = clean_all[(clean_all['RT2'] >= 1.5)| (clean_all['RT2'] <= 0.05)]
# Clean the dataset from RTs > 1.50 s in the secondary task (not < 0.05 because wrong answers = 0)
secondary = clean_all[clean_all['RT2'] <= 1.5]

# Create different datasets for each trial type at the secondary task
test_inhibition_Go_all = secondary[secondary['Type'] == "GO"]
test_inhibition_Miss_all = secondary[secondary['Type'] == "Miss"]
test_inhibition_NoGo_all = secondary[secondary['Type'] == "NO-GO"]
test_inhibition_FA_all = secondary[secondary['Type'] == "False Alarm"]

# Clean RTs < 0.05 s for the Go trials
test_inhibition_Go_all = test_inhibition_Go_all[test_inhibition_Go_all['RT2'] >= 0.05]

# Calculate mean RTs for each trial type at the Stroop task for each participant
inhib_meanRTs_all = test_inhibition_Go_all.groupby(["P", "Congruence"]).agg({
    'RT2': 'mean', 
    'Group': 'first'  # Keeps the first occurrence within each group
}).reset_index()

# Calculate overall mean RTs for each participant 
inhib_meanRTs_participant = test_inhibition_Go_all.groupby(["P"])
inhib_meanRTs_participant = inhib_meanRTs_participant['RT2'].mean() #*1000
inhib_meanRTs_participant = inhib_meanRTs_participant.reset_index()

# Calculate and print on-screen overall mean RTs at the secondary task for each Stroop trial type
inhib_meanRTs_incongr_all = inhib_meanRTs_all[inhib_meanRTs_all['Congruence'] == 'Incongruent']['RT2']
inhib_meanRTs_congr_all = inhib_meanRTs_all[inhib_meanRTs_all['Congruence'] == 'Congruent']['RT2']
inhib_meanRTs_neutral_all = inhib_meanRTs_all[inhib_meanRTs_all['Congruence'] == 'Neutral']['RT2']

meanRT_after_congr_all = inhib_meanRTs_congr_all.mean() #*1000
stdRT_after_congr_all = inhib_meanRTs_congr_all.std() #*1000
print("\nRTs after Stroop congruent trials:", meanRT_after_congr_all, "±", stdRT_after_congr_all)

meanRT_after_incongr_all = inhib_meanRTs_incongr_all.mean() #*1000
stdRT_after_incongr_all = inhib_meanRTs_incongr_all.std()#*1000
print("\nRTs after Stroop incongruent trials:", meanRT_after_incongr_all, "±", stdRT_after_incongr_all)

meanRT_after_neutral_all = inhib_meanRTs_neutral_all.mean() #*1000
stdRT_after_neutral_all = inhib_meanRTs_neutral_all.std() #*1000
print("\nRTs after Stroop neutral trials:", meanRT_after_neutral_all, "±", stdRT_after_neutral_all)

# Perform registerd two-tailed paired t-tests (p < 0.05)
t_incong_congRT_all = ttest_rel(inhib_meanRTs_incongr_all, inhib_meanRTs_congr_all, alternative='two-sided')

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_incong_congRT_all = cohen_d(inhib_meanRTs_incongr_all, inhib_meanRTs_congr_all)

# Print t-test and Cohen's d results
print("REGISTERED DATA ANALYSIS: Two-tailed paired t-test (p < 0.05)")
print("After Incongruent vs. After Congruent:")
print("t-statistic:", t_incong_congRT_all.statistic)
print("p-value:", t_incong_congRT_all.pvalue)
print("Cohen's d:", d_incong_congRT_all)

# Perform registered two-tailed paired t-test (p < 0.05) to compare the RTs after incongruent and congruent Stroop trials
t_incong_neutralRT = ttest_rel(inhib_meanRTs_incongr_all, inhib_meanRTs_neutral_all, alternative='two-sided')
t_cong_neutralRT = ttest_rel(inhib_meanRTs_congr_all, inhib_meanRTs_neutral_all, alternative='two-sided')

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_incong_neutralRT = cohen_d(inhib_meanRTs_incongr_all, inhib_meanRTs_neutral_all)
d_cong_neutralRT = cohen_d(inhib_meanRTs_congr_all, inhib_meanRTs_neutral_all)

# Print t-test and Cohen's d results
print("EXPLORATORY: T-tests comparing the mean RTs after Stroop trial types.")
print("\nAfter Incongruent vs. After Neutral:")
print("t-statistic:", t_incong_neutralRT.statistic)
print("p-value:", t_incong_neutralRT.pvalue)
print("Cohen's d:", d_incong_neutralRT)
print("\nAfter Congruent vs. After Neutral:")
print("t-statistic:", t_cong_neutralRT.statistic)
print("p-value:", t_cong_neutralRT.pvalue)
print("Cohen's d:", d_cong_neutralRT, "\n")

#------------------------ SAVE RESULTS FILE
# Store file in the output folder
resultspath = 'GitHub/data/' # select the proper datapath
# Specify the output file path
excel_output_path = os.path.join(resultspath, 'exp2_secondary_meanRTs.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    inhib_meanRTs_all.to_excel(writer, sheet_name='exp2_secondary_meanRTs', index=False)
print(f"Results saved to {excel_output_path}")

###################### ANOVA
# Perform one-way ANOVA with Stroop effect x 3 Experiments
model_secondary = ols('RT2 ~ C(Congruence)', data=inhib_meanRTs_all).fit()
anova_table_secondary = sm.stats.anova_lm(model_secondary, typ=2)
print(anova_table_secondary)
model_results = model_secondary.summary()
print(model_results)

tukey_oneway = pairwise_tukeyhsd(endog = inhib_meanRTs_all["RT2"], groups = inhib_meanRTs_all["Congruence"])
# Display the results
print(tukey_oneway.summary())

# Calculate Eta-Squared
SSB = anova_table_secondary['sum_sq']['C(Congruence)']  # Sum of Squares Between Groups
SST = anova_table_secondary['sum_sq'].sum()                 # Total Sum of Squares (Between + Within)
eta_squared = SSB / SST

print(f"Eta-Squared (η²): {eta_squared}\n")

#----------------------------------------------
# Analyse the data for the two groups separately
# Group 1 = 240 GO, 120 NO-GO
df_group1 = clean_all[clean_all["Group"] == 1].copy()
print("Analysis for Group 1: 240 GO, 120 NO-GO")
# # Group 2 = 200 GO, 160 NO-GO
df_group2 = clean_all[clean_all["Group"] == 2].copy()
print("Analysis for Group 2: 200 GO, 160 NO-GO")

# Clean RTs for each group 
clean_group = df_group1[df_group1['RT2'] >= 0.05].copy() # CHANGE THE GROUP NUMBER
# results_group1 = calculate_rt_stats(clean_group)

# Create different datasets for each trial type at the secondary task for the selected group
test_inhibition_Go = clean_group[clean_group['Type'] == "GO"]
test_inhibition_Miss = clean_group[clean_group['Type'] == "Miss"]
test_inhibition_NoGo = clean_group[clean_group['Type'] == "NO-GO"]
test_inhibition_FA = clean_group[clean_group['Type'] == "False Alarm"]

# Store the excluded RTs in a different dataset
excluded_go = test_inhibition_Go[(test_inhibition_Go['RT2'] >= 1.500) | (test_inhibition_Go['RT2'] <= 0.05)]
# Clean the RTs in the main dataset
test_inhibition_Go = test_inhibition_Go[(test_inhibition_Go['RT2'] <= 1.500) & (test_inhibition_Go['RT2'] >= 0.05)]

# Calculate the mean RTs for all participants
inhib_meanRTs_All = test_inhibition_Go.groupby(["P"])
inhib_meanRTs_All = inhib_meanRTs_All['RT2'].mean() #*1000
inhib_meanRTs_All = inhib_meanRTs_All.reset_index()

# Calculate the mean RTs for all participants and Stroop trial types
inhib_meanRTs = test_inhibition_Go.groupby(["P","Congruence"])
inhib_meanRTs = inhib_meanRTs['RT2'].mean() #*1000
inhib_meanRTs = inhib_meanRTs.reset_index()

# Calculate the mean RTs for the different Stroop trial types
inhib_meanRTs_trialtype = test_inhibition_Go.groupby(["Congruence"])
inhib_meanRTs_trialtype = inhib_meanRTs_trialtype['RT2'].mean()#*1000
inhib_meanRTs_trialtype = inhib_meanRTs_trialtype.reset_index()

# Calculate and print on-screen overall mean RTs at the secondary task
overallmeanRTs = test_inhibition_Go['RT2'].mean()#*1000
overallstdRTs = test_inhibition_Go['RT2'].std()#*1000
print("\nRTs to Go trials overall:", overallmeanRTs, "±", overallstdRTs)

# Calculate and print on-screen overall mean RTs at the secondary task for each Stroop trial type
inhib_meanRTs_incongr = inhib_meanRTs[inhib_meanRTs['Congruence'] == 'Incongruent']['RT2']
inhib_meanRTs_congr = inhib_meanRTs[inhib_meanRTs['Congruence'] == 'Congruent']['RT2']
inhib_meanRTs_neutral = inhib_meanRTs[inhib_meanRTs['Congruence'] == 'Neutral']['RT2']

meanRT_after_congr = inhib_meanRTs_congr.mean() #*1000
stdRT_after_congr = inhib_meanRTs_congr.std() #*1000
print("\nRTs to Go trials after congruent:", meanRT_after_congr, "±", stdRT_after_congr)

meanRT_after_incongr = inhib_meanRTs_incongr.mean() #*1000
stdRT_after_incongr = inhib_meanRTs_incongr.std()#*1000
print("\nRTs to Go trials after incongruent:", meanRT_after_incongr, "±", stdRT_after_incongr)

meanRT_after_neutral = inhib_meanRTs_neutral.mean() #*1000
stdRT_after_neutral = inhib_meanRTs_neutral.std() #*1000
print("\nRTs to Go trials after neutral:", meanRT_after_neutral, "±", stdRT_after_neutral)

###################### ANOVA
# Perform one-way ANOVA with Stroop effect x 3 Experiments
model_secondary = ols('RT2 ~ C(Congruence)', data=inhib_meanRTs).fit()
anova_table_secondary = sm.stats.anova_lm(model_secondary, typ=2)
print(anova_table_secondary)

# Calculate Eta-Squared
SSB = anova_table_secondary['sum_sq']['C(Congruence)']  # Sum of Squares Between Groups
SST = anova_table_secondary['sum_sq'].sum()                 # Total Sum of Squares (Between + Within)
eta_squared = SSB / SST

print(f"Eta-Squared (η²): {eta_squared}\n")



# Perform registered two-tailed paired t-test (p < 0.05) to compare the RTs after incongruent and congruent Stroop trials
t_incong_congRT = ttest_rel(inhib_meanRTs_incongr, inhib_meanRTs_congr, alternative='two-sided')
t_incong_neutralRT = ttest_rel(inhib_meanRTs_incongr, inhib_meanRTs_neutral, alternative='two-sided')
t_cong_neutralRT = ttest_rel(inhib_meanRTs_congr, inhib_meanRTs_neutral, alternative='two-sided')

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_incong_congRT = cohen_d(inhib_meanRTs_incongr, inhib_meanRTs_congr)
d_incong_neutralRT = cohen_d(inhib_meanRTs_incongr, inhib_meanRTs_neutral)
d_cong_neutralRT = cohen_d(inhib_meanRTs_congr, inhib_meanRTs_neutral)

# Print t-test and Cohen's d results
print("EXPLORATORY: T-tests comparing the mean RTs after Stroop trial types for the two groups separately.")
print("After Incongruent vs. After Congruent:")
print("t-statistic:", t_incong_congRT.statistic)
print("p-value:", t_incong_congRT.pvalue)
print("Cohen's d:", d_incong_congRT)
print("\nAfter Incongruent vs. After Neutral:")
print("t-statistic:", t_incong_neutralRT.statistic)
print("p-value:", t_incong_neutralRT.pvalue)
print("Cohen's d:", d_incong_neutralRT)
print("\nAfter Congruent vs. After Neutral:")
print("t-statistic:", t_cong_neutralRT.statistic)
print("p-value:", t_cong_neutralRT.pvalue)
print("Cohen's d:", d_cong_neutralRT, "\n")


#---------------------------------------- SAVE RESULTS FILE
# Store file in the output folder
resultspath = '' # select the proper datapath

# Clean datasets to perform exploratory analyses: Stroop task
excel_output_path = os.path.join(resultspath, 'exp2_stroop_clean.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    clean_all.to_excel(writer, sheet_name='exp2_stroop_clean', index=False)
print(f"Results saved to {excel_output_path}")

# Clean datasets to perform exploratory analyses: secondary task
excel_output_path = os.path.join(resultspath, 'exp2_secondary_clean.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    secondary.to_excel(writer, sheet_name='exp2_secondary_clean', index=False)
print(f"Results saved to {excel_output_path}")

# Stroop mean RTs for plots
excel_output_path = os.path.join(resultspath, 'exp2_stroop_meanRTs.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    stroop_meanRTs.to_excel(writer, sheet_name='exp2_stroop_meanRTs', index=False)
print(f"Results saved to {excel_output_path}")
