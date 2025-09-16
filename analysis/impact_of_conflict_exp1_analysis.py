# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:41:33 2024

@author: Marta La Pietra
Project: The "sweet spot" of Cognitive Conflict (PID2020-114717RA-I00)
Description: This script analyzes the data from the Experiment 1 (Stroop task intermixed with a speeded detection task) in the Registered Report titled "Exploring the impact of cognitive conflict on subsequent cognitive processes". 
It calculates the mean reaction times (RTs) and accuracies for the Stroop task and the RTs for the speeded detection task. 
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

filename = "exp1_all_trials.csv"
file_path = os.path.join(datapath, foldername, filename)
full_data = pd.read_csv(file_path)

congruency_mapping = {1: 'Congruent', 0: 'Incongruent', 2: 'Neutral'}
full_data['Congruence'] = full_data['Congruency'].map(congruency_mapping)

# Identify the first trial of each block
full_data['Excluded'] = (full_data['N'] == 1).astype(int)
first_trials = [1 + 36 * (i - 1) for i in range(1, 11)]
# Assign 1 to 'excluded' column for these trials
full_data['Excluded'] = full_data['N'].apply(lambda x: 1 if x in first_trials else 0)

# Save first trials of each block in the Stroop task
excluded_first = full_data[full_data['Excluded'] != 0].copy()
# And exclude them 
df = full_data[full_data['Excluded'] == 0].copy()    

# Save too slow RTs to the Stroop task
excluded_tooslow = full_data[full_data['RT'] > 1.5].copy()
# And exclude them 
df = df[df['RT'] < 1.5].copy()    

# Save too fast RTs to the Stroop task
excluded_toofast = df[df['RT']<= 0.05].copy()
# And exclude them 
df = df[df['RT'] >= 0.05].copy() 

# Overall accuracy for the Stroop task
overall_acc = df['Accuracy'].mean() * 100
overall_acc_std = df['Accuracy'].std() * 100
print("\nOverall Accuracy at the Stroop task in Experiment 1:", overall_acc, "±", overall_acc_std)

# Remove erroneous answers (Accuracy = 0) from the DataFrame
clean_df = df[df['Accuracy'] != 0].copy()

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

# Calculate mean RTs for each trial type and participant
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

# -------------------------------------------------------
# SECONDARY TASK
# Create a dataframe just when the secondary task occurred
data_secondary = clean_df[clean_df['Orientation'] != 0].copy()
# Too slow
excluded_tooslow_secondary = data_secondary[data_secondary['RT2'] > 1.5].copy()
# And exclude them 
data_secondary = data_secondary[data_secondary['RT2'] < 1.5].copy()    
#If there are, save those:
excluded_toofast_secondary = data_secondary[data_secondary['RT2']<= 0.05].copy()
# And exclude them 
data_secondary = data_secondary[data_secondary['RT2'] >= 0.05].copy() 

# Store the mean RTs at the secondary task for each participants and trial type
detection_meanRTs = data_secondary.groupby(["P","Congruence"])
detection_meanRTs = detection_meanRTs['RT2'].mean()#*1000
detection_meanRTs = detection_meanRTs.reset_index()

# Calculate mean RTs at the secondary task after each Stroop trial type
def calculate_rt_stats_secondary(df):
    # Create a list to store the results
    results = []

    # Group the DataFrame by Participant Number
    grouped = df.groupby('P')

    for participant, data in grouped:
        # Calculate mean RTs for each Congruency condition
        overallmeanRTs = data['RT2'].mean()
        mean_rt_after_congruent = data[data['Congruency'] == 1]['RT2'].mean()
        mean_rt_after_incongruent = data[data['Congruency'] == 0]['RT2'].mean()
        mean_rt_after_neutral = data[data['Congruency'] == 2]['RT2'].mean()
                
        # Append the results to the list
        results.append({
            'Participant': participant,
            'Overall_Mean_RTs': overallmeanRTs,
            'Mean_RTs_Congruent': mean_rt_after_congruent,
            'Mean_RTs_Incongruent': mean_rt_after_incongruent,
            'Mean_RTs_Neutral': mean_rt_after_neutral,
        })

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Example usage:
results_secondary_df = calculate_rt_stats_secondary(data_secondary)

# Mean RTs for the secondary task after each Stroop trial type and participant
secondary_overall_rts = results_secondary_df['Overall_Mean_RTs']
secondary_congruent_rts = results_secondary_df['Mean_RTs_Congruent']
secondary_incongruent_rts = results_secondary_df['Mean_RTs_Incongruent']
secondary_neutral_rts = results_secondary_df['Mean_RTs_Neutral']

# MEANS
secondary_overall_rts_mean = secondary_overall_rts.mean() *1000
secondary_overall_rts_sd = secondary_overall_rts.std()*1000
secondary_congruent_rts_mean = secondary_congruent_rts.mean() *1000
secondary_congruent_rts_sd = secondary_congruent_rts.std()*1000
secondary_incongruent_rts_mean = secondary_incongruent_rts.mean()*1000
secondary_incongruent_rts_sd = secondary_incongruent_rts.std()*1000
secondary_neutral_rts_mean = secondary_neutral_rts.mean()*1000
secondary_neutral_rts_sd = secondary_neutral_rts.std()*1000

print("\nAnalysis of the responses to the Speeded Detection Task:")
print(f"Overall Mean RTs: {secondary_overall_rts_mean:.2f} ± {secondary_overall_rts_sd:.2f} ms")
print(f"Overall Mean RTs after CONGRUENT Trials: {secondary_congruent_rts_mean:.2f} ± {secondary_congruent_rts_sd:.2f} ms")
print(f"Overall Mean RTs after INCONGRUENT Trials: {secondary_incongruent_rts_mean:.2f} ± {secondary_incongruent_rts_sd:.2f} ms")
print(f"Overall Mean RTs after NEUTRAL Trials: {secondary_neutral_rts_mean:.2f} ± {secondary_neutral_rts_sd:.2f} ms\n")

# Perform registered two-tailed paired t-test (p < 0.05) to compare the RTs after incongruent and congruent Stroop trials
t_incong_congRT = ttest_rel(secondary_incongruent_rts, secondary_congruent_rts, alternative='two-sided')
t_incong_neutralRT = ttest_rel(secondary_incongruent_rts, secondary_neutral_rts, alternative='two-sided')
t_cong_neutralRT = ttest_rel(secondary_congruent_rts, secondary_neutral_rts, alternative='two-sided')

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
    return diff_mean / pooled_std

# Calculate Cohen's d
d_incong_congRT = cohen_d(secondary_incongruent_rts, secondary_congruent_rts)
d_incong_neutralRT = cohen_d(secondary_incongruent_rts, secondary_neutral_rts)
d_cong_neutralRT = cohen_d(secondary_congruent_rts, secondary_neutral_rts)

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

###################### ANOVA
# Perform one-way ANOVA with Stroop effect x 3 Experiments
model_secondary = ols('RT2 ~ C(Congruence)', data=detection_meanRTs).fit()
anova_table_secondary = sm.stats.anova_lm(model_secondary, typ=2)
print(anova_table_secondary)
model_results = model_secondary.summary()
print(model_results)
tukey_oneway = pairwise_tukeyhsd(endog = detection_meanRTs["RT2"], groups = detection_meanRTs["Congruence"])
# Display the results
print(tukey_oneway.summary())

# Calculate Eta-Squared
SSB = anova_table_secondary['sum_sq']['C(Congruence)']  # Sum of Squares Between Groups
SST = anova_table_secondary['sum_sq'].sum()                 # Total Sum of Squares (Between + Within)
eta_squared = SSB / SST

print(f"Eta-Squared (η²): {eta_squared}\n")


#----------------------------------------------- SAVE RESULTS FILE
resultspath = 'GitHub/data/' # select the proper datapath

# Clean datasets to perform exploratory analyses: Stroop task
excel_output_path = os.path.join(resultspath, 'exp1_stroop_clean.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    clean_df.to_excel(writer, sheet_name='exp1_stroop_clean', index=False)
print(f"Results saved to {excel_output_path}")

# Clean datasets to perform exploratory analyses: secondary task
excel_output_path = os.path.join(resultspath, 'exp1_secondary_clean.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    data_secondary.to_excel(writer, sheet_name='exp1_secondary_clean', index=False)
print(f"Results saved to {excel_output_path}")

# Stroop mean RTs for plots
excel_output_path = os.path.join(resultspath, 'exp1_stroop_meanRTs.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    stroop_meanRTs.to_excel(writer, sheet_name='exp1_stroop_meanRTs', index=False)
print(f"Results saved to {excel_output_path}")

# Secondary mean RTs for plots
excel_output_path = os.path.join(resultspath, 'exp1_secondary_meanRTs.xlsx')
# Save the DataFrame to an Excel file in a single sheet
with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    detection_meanRTs.to_excel(writer, sheet_name='exp1_secondary_meanRTs', index=False)
print(f"Results saved to {excel_output_path}")

