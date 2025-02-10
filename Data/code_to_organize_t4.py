import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import ttest_ind

"""
Name: Abdikarim Jimale
Date: 02-09-2025
"""

# Load the data
t1_data = pd.read_csv('Data/t1_user_active_min.csv')
t2_data = pd.read_csv('Data/t2_user_variant.csv')
t3_data = pd.read_csv('Data/t3_user_active_min_pre.csv')
t4_data = pd.read_csv('Data/t4_user_attributes.csv')

# Merge data
full_data = t4_data.merge(t2_data, on='uid').merge(t3_data, on='uid', how='left')

# Group data by user type and gender and active minutes.
group_data = full_data.groupby(['user_type', 'gender'])['active_mins'].mean().unstack()

# Plotting
group_data.plot(kind='bar')
plt.title('Average active minutes by user type and gender')
plt.xlabel('User type')
plt.ylabel('Average active minutes')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()