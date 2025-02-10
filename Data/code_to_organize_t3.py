# Import necessary libraries
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
Name: Abdikarim Jimale
Date: 02-09-2025
"""

# Load the data from CSV files
t3_data = pd.read_csv('Data/t3_user_active_min_pre.csv')
t2_data = pd.read_csv('Data/t2_user_variant.csv')

# Aggregate total active minutes per user in t3 (After 2019-02-05)
t3_total_active_mins = t3_data.groupby('uid')['active_mins'].sum().reset_index()


# Merge t2 data with aggregated active minutes data from t3
mergerd_data = t2_data.merge(t3_total_active_mins.rename(columns={'active_mins': 'active_mins_t3'}), on='uid', how='left')

# Handle missing values by filling NA with 0 in the active minutes colum
mergerd_data['active_mins'] = mergerd_data['active_mins_t3'].fillna(0)

# Select only the necessary columns: 'uid', 'variant_number' and 'active_mins'
t3_final = mergerd_data[['uid', 'variant_number', 'active_mins']]

# Save the cleaned and merged data to a new CSV file for further analysis
t3_final.to_csv('/home/abdikarim/hw3-ajimale/Data/t3_organized_data.csv', index=False)
print("Data is organized and save.")

# Split data into two group based on variant_number: Control (group A) and Treatment (Group B)
group_A = t3_final[t3_final['variant_number'] == 0]['active_mins'].dropna() #Control group (group 01)
group_B = t3_final[t3_final['variant_number'] == 1]['active_mins'].dropna() #Tretment group (group 02)

# Print the number of rows in the final dataframe.
print("The number of element or rows at t3 is: ", len(t3_final))

# Calculate and print the mean and median for both groups.
mean_grop_A = group_A.mean()
median_group_A = group_A.median()
mean_grop_B = group_B.mean()
median_group_B = group_B.median()

print("*" * 100)
print("group A size: ", len(group_A))
print("group B size: ", len(group_B))
print("Mean of active minutes for group A: ", mean_grop_A)
print("Median of active minutes for group A: ", median_group_A)
print("Mean of active minutes for group B: ", mean_grop_B)
print("Median of active minutes for group B: ", median_group_B)

# Preform a t-test to compare the means of the two groups
print("*" * 100)
t_stat, p_value = stats.ttest_ind(group_A, group_B)
print("T-statistic: ", t_stat)
print("P-vlaue: ", p_value)

# Interpret the p-vlaue from the t-test
if p_value < 0.05:
    print("There is a statically difference between group 1 and group 2")
    print("Conclusion: There difference in active minute between the two group.")
else: 
    print("There is no statically difference between group 1 and group 2")
    print("Conclusion: Thers no different in active minute between the two group.")

# Part 04: Q5 to Q6 
max_active_minutes = t3_final['active_mins'].max()
print("*" * 100)
print("MAximum active minutes recorded: ", max_active_minutes)

# calculate Q1 and Q3
Q1 = t3_final['active_mins'].quantile(0.25)
Q3 = t3_final['active_mins'].quantile(0.75)
IQR = Q3 - Q1

# calculate the minimum and maximum  values within the whisker range
whisker_low = Q1 - 1.5 * IQR
whisker_high = Q3 + 1.5 * IQR 

print("Whisker low (minimum in bixplot): ", whisker_low)
print("Whisker high (maximum in bixplot): ", whisker_high)

# Define a funcation to remove outliers
def remove_outliers(series, column_name):
    # calculate Q1 and Q3
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # calculate the minimum and maximum  values within the whisker range
    low_bounnd = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR 

    filtered_df = t3_final[(t3_final[column_name] >= low_bounnd) & (t3_final[column_name] <= upper_bound)]
    return filtered_df

# Apply the funcation to each group
group_A_without_outliers = remove_outliers(group_A, 'active_mins') 
group_B_without_outliers = remove_outliers(group_B, 'active_mins') 
total_without_outliers = pd.concat([group_A_without_outliers, group_B_without_outliers], ignore_index=True)

print("*" * 100)
print("Filtered Group A size after remove outliner: ", len(group_A_without_outliers))
print("Filtered Group B size after remove outliner: ", len(group_B_without_outliers))
print("Filtered Group total size after remove outliner: ", len(total_without_outliers))

# Part 04 Q 07
# Calculate and print the mean and median for both groups.
mean_grop_A_outliners = group_A_without_outliers['active_mins'].mean()
median_group_A_outliners  = group_A_without_outliers['active_mins'].median()
mean_grop_B_outliners  = group_B_without_outliers['active_mins'].mean()
median_group_B_outliners  = group_B_without_outliers['active_mins'].median()

print("*" * 100)
print("Mean of active minutes for group A without outliners : ", mean_grop_A_outliners)
print("Median of active minutes for group A without outliners: ", median_group_A_outliners)
print("Mean of active minutes for group B without outliners: ", mean_grop_B_outliners)
print("Median of active minutes for group B without outliners: ", median_group_B_outliners)

# Preform a t-test to compare the means of the two groups
print("*" * 100)
t_stat, p_value = stats.ttest_ind(group_A_without_outliers['active_mins'], group_B_without_outliers['active_mins'])
print("T-statistic: ", t_stat)
print("P-vlaue: ", p_value)

# Interpret the p-vlaue from the t-test
if p_value < 0.05:
    print("There is a statically difference between group 1 and group 2")
    print("Conclusion: There difference in active minute between the two group.")
else: 
    print("There is no statically difference between group 1 and group 2")
    print("Conclusion: Thers no different in active minute between the two group.")

print("*" * 100)


# Part 03 and part 04 - Data Visualization with outliner
sns.set(style='whitegrid')

#***************************************** outliners & without outliners ***********************************

# Apply log transformation to 'active_mins' for better distribution ***** with outliners ****
t3_final['active_mins'] = np.log1p(t3_final['active_mins'])

# # Apply log transformation to 'active_mins' for better distribution ***** without outliners ****
# t3_final['active_mins'] = np.log1p(total_without_outliers['active_mins'])

# Create histograms for each group
plt.figure(figsize=(14,8))

# Group 01 (Control group)
plt.subplot(2, 2, 1)
sns.histplot(data= t3_final[t3_final['variant_number'] == 0], x='active_mins', kde=True, color='blue')
plt.title('Histogram of Active minutes for control group (Group 01) \n with outliner ')
plt.xlabel('Active Minutes')
plt.ylabel('Frequency')

# Group 02 (Treatment group)
plt.subplot(2, 2, 2)
sns.histplot(data= t3_final[t3_final['variant_number'] == 1], x='active_mins', kde=True, color='orange')
plt.title('Histogram of Active minutes for Tretment group (Group 02) \n with outliner')
plt.xlabel('Active Minutes')
plt.ylabel('Frequency')

# Total group (combined)
plt.subplot(2,2,3)
sns.histplot(t3_final['active_mins'], kde=True, color='green')
plt.title('Histogram of Active minutes for All users \n with outliner')
plt.xlabel('Active Minutes')
plt.ylabel('Frequency')

# Lobel the goups for better understanding in the plot
t3_final['group_label'] = t3_final['variant_number'].replace({0: 'Group 1 (Control)', 1:'Group 2 (Treatment)'})

#Create a combined dataframe with a 'Total' group for boxplot
t2_total = t3_final.copy()
t2_total['group_label'] = 'Total'
t2_combined = pd.concat([t3_final, t2_total])

# Create a boxplot of log-transformed active minutes for all groups  
plt.figure(figsize=(8, 6))
sns.boxplot(data=t2_combined, x='group_label', y='active_mins', palette=['blue','orange','green'])
plt.title('Boxplote of Log-transformed Active Minutes for Groups A, B, and total \n with outliner')
plt.xlabel('Group')
plt.ylabel('Log(Active Minutes)')
plt.show()

