
####################################################################
# Title: synoptic.py
# Author: Michelle Hughes
# Aim: Creation of dummy data for synoptic project looking at the
# impact of activity on anxiety levels
####################################################################

####################################################################
# Package management
####################################################################
import sys
import pandas as pd
from tabulate import tabulate
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

####################################################################
# Data creation
####################################################################
# Define the number of participants
num_participants = 30

# Create dummy data
np.random.seed(54)  # For reproducibility

data_control = {
    "ID": range(1, num_participants + 1),
    "Age": np.random.randint(45, 56, num_participants),
    "Waist_Circumference_Baseline": np.random.normal(96, 4, num_participants),
    "Sub_75_Completed": np.where(np.random.choice([True, False], num_participants, p=[0.75, 0.25]), 0, 1),
    "PCr_ATP_Baseline": np.random.uniform(1.2, 1.5, num_participants),
    "GAD7_Baseline": np.random.choice([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], num_participants, 
        p=[.12,.12,.12,.12,.12,0.06,0.06,0.06,0.06,0.06,0.015,0.015,0.014,0.014,0.014,0.014,0.014])
}

data_intervention = {
    "ID": range(1, num_participants + 1),
    "Age": np.random.randint(45, 56, num_participants),
    "Waist_Circumference_Baseline": np.random.normal(96, 4, num_participants),
    "Sub_75_Completed": np.where(np.random.choice([True, False], num_participants, p=[0.75, 0.25]), 0, 1),
    "PCr_ATP_Baseline": np.random.uniform(1.2, 1.5, num_participants),
    "GAD7_Baseline": np.random.choice([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], num_participants, 
        p=[.12,.12,.12,.12,.12,0.06,0.06,0.06,0.06,0.06,0.015,0.015,0.014,0.014,0.014,0.014,0.014])
}

# Convert to DataFrame
df1 = pd.DataFrame(data_control)
df2 = pd.DataFrame(data_intervention)
df1['Group'] = 'Control'
df2['Group'] = 'Intervention'

df = pd.concat([df1, df2], axis=0, ignore_index=True)

df['PCr_ATP_Week_15'] = df['PCr_ATP_Baseline']
df['GAD7_Week_15'] = df['GAD7_Baseline']
df['Waist_Circumference_Week_15'] = df['Waist_Circumference_Baseline']

# Adjust the PCr/ATP values for Intervention group assuming some positive change
df.loc[df['Group'] == 'Intervention', 'PCr_ATP_Week_15'] += np.random.normal(0.1, 0.01, len(df[df['Group'] == 'Intervention']))
# Adjust the PCr/ATP values for Control group assuming some neutral change
df.loc[df['Group'] == 'Control', 'PCr_ATP_Week_15'] += np.random.normal(0, 0.01, len(df[df['Group'] == 'Control']))

df['temp'] = np.random.choice([-1,0,1,2,3,4], num_participants*2, p=[.02,.08,.18,.4,.22,.1])
# Adjust the GAD7 scores for Intervention group assuming some improvement
df.loc[df['Group'] == 'Intervention', 'GAD7_Week_15'] -= df['temp']
df = df.drop(columns=['temp'])
# Adjust the GAD7 scores for Control group assuming no improvement but some change
df.loc[df['Group'] == 'Control', 'GAD7_Week_15'] -= np.random.randint(-1, 2, len(df[df['Group'] == 'Control']))

# Adjust the Waist size for Intervention group assuming some improvement
df.loc[df['Group'] == 'Intervention', 'Waist_Circumference_Week_15'] -= np.random.normal(2, 1, len(df[df['Group'] == 'Intervention']))
# Adjust the Waist size for Control group assuming no improvement but some change
df.loc[df['Group'] == 'Control', 'Waist_Circumference_Week_15'] -= np.random.normal(0, 1, len(df[df['Group'] == 'Control']))

# Set an upper limit of 21 for GAD7_Week_15
df['GAD7_Week_15'] = df['GAD7_Week_15'].clip(upper=21)

# Set Sub_75_Completed to equal 0 for all participants in Control
df.loc[df['Group'] == 'Control', 'Sub_75_Completed'] = 0

# Round columns appropriately
df['Waist_Circumference_Baseline'] = df['Waist_Circumference_Baseline'].round(0).astype(int)
df['Waist_Circumference_Week_15'] = df['Waist_Circumference_Week_15'].round(0).astype(int)
df['PCr_ATP_Baseline'] = df['PCr_ATP_Baseline'].round(2)
df['PCr_ATP_Week_15'] = df['PCr_ATP_Week_15'].round(2)
'''
####################################################################
# Visualise table
####################################################################

class TableWindow(QMainWindow):
    def __init__(self, dataframe):
        super().__init__()
        self.setWindowTitle("Table Display")
        self.setGeometry(100, 100, 600, 400)
        
        self.create_table(dataframe)
    
    def create_table(self, dataframe):
        self.table_widget = QTableWidget()
        self.setCentralWidget(self.table_widget)
        
        self.table_widget.setRowCount(dataframe.shape[0])
        self.table_widget.setColumnCount(dataframe.shape[1])
        self.table_widget.setHorizontalHeaderLabels(dataframe.columns)
        
        for i in range(dataframe.shape[0]):
            for j in range(dataframe.shape[1]):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(dataframe.iat[i, j])))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TableWindow(df)
    window.show()
    sys.exit(app.exec_())
'''
df['GAD7_Difference'] = df['GAD7_Week_15'] - df['GAD7_Baseline']
df['PCr_ATP_Difference'] = df['PCr_ATP_Week_15'] - df['PCr_ATP_Baseline']

####################################################################
# Means and standard deviations
####################################################################
'''
# Analyse data
print("Average age by group")
print(df.groupby('Group')['Age'].mean())
print("Standard deviation of age by group")
print(df.groupby('Group')['Age'].std())

print("Average baseline PCr/ATP by group")
print(df.groupby('Group')['PCr_ATP_Baseline'].mean())
print("Standard deviation of baseline PCr/ATP by group")
print(df.groupby('Group')['PCr_ATP_Baseline'].std())
print("Average week 15 PCr/ATP by group")
print(df.groupby('Group')['PCr_ATP_Week_15'].mean())

print("Average starting GAD by group")
print(df.groupby('Group')['GAD7_Baseline'].mean())
print("Standard deviation of starting GAD by group")
print(df.groupby('Group')['GAD7_Baseline'].std())
print("Average week 15 GAD by group")
print(df.groupby('Group')['GAD7_Week_15'].mean())

print("Average starting waist by group")
print(df.groupby('Group')['Waist_Circumference_Baseline'].mean())
print("Standard deviation of starting waist by group")
print(df.groupby('Group')['Waist_Circumference_Baseline'].std())
print("Average week 15 waist by group")
print(df.groupby('Group')['Waist_Circumference_Week_15'].mean())

print("Sub 75 completion by group")
print(df.groupby('Group')['Sub_75_Completed'].mean())
'''
####################################################################
# Box and whisker charts
####################################################################
'''
# Melt data to create PCr chart
df_melted_ATP = df.melt(id_vars=['Group'], value_vars=['PCr_ATP_Baseline', 'PCr_ATP_Week_15'], var_name='Measure', value_name='PCr_ATP')
df_melted_ATP['Measure'] = df_melted_ATP['Measure'].replace('PCr_ATP_Baseline', ' Baseline', regex=True)
df_melted_ATP['Measure'] = df_melted_ATP['Measure'].replace('PCr_ATP_Week_15', '15 weeks', regex=True)

# Create a box and whisker plot of column A split by column B
color_palette = ['#999999', '#666666']
plt.figure(figsize=(10, 6))
bp = df_melted_ATP.boxplot(column='PCr_ATP', by=['Group','Measure'], ax=plt.gca(), patch_artist=True, widths=0.6)

# Set titles and labels
plt.title('')
plt.suptitle('')  # Suppress the automatic title from pandas
plt.xlabel('')
plt.ylabel('PCr/ATP', color='#000000', fontsize=20)
custom_labels = ['Control\nBaseline', 'Control\n15 weeks', 'Intervention\nBaseline', 'Intervention\n15 weeks']
plt.xticks(ticks=[1, 2, 3, 4], labels=custom_labels, fontsize=20)
plt.yticks(fontsize=20)

# Set background color to white
plt.gca().set_facecolor('white')

# Set spines color to gray
plt.gca().spines['top'].set_color('#888888')
plt.gca().spines['right'].set_color('#888888')
plt.gca().spines['bottom'].set_color('#888888')
plt.gca().spines['left'].set_color('#888888')

# Show the plot
plt.tight_layout()
plt.show()
'''
'''
# Melt data to create GAD7 chart
df_melted_ATP = df.melt(id_vars=['Group'], value_vars=['GAD7_Baseline', 'GAD7_Week_15'], var_name='Measure', value_name='GAD7')
df_melted_ATP['Measure'] = df_melted_ATP['Measure'].replace('GAD7_Baseline', ' Baseline', regex=True)
df_melted_ATP['Measure'] = df_melted_ATP['Measure'].replace('GAD7_Week_15', '15 weeks', regex=True)

# Create a box and whisker plot of column A split by column B
color_palette = ['#999999', '#666666']
plt.figure(figsize=(10, 6))
bp = df_melted_ATP.boxplot(column='GAD7', by=['Group','Measure'], ax=plt.gca(), patch_artist=True, widths=0.6)

# Set titles and labels
plt.title('')
plt.suptitle('')  # Suppress the automatic title from pandas
plt.xlabel('')
plt.ylabel('GAD7', color='#000000', fontsize=20)
custom_labels = ['Control\nBaseline', 'Control\n15 weeks', 'Intervention\nBaseline', 'Intervention\n15 weeks']
plt.xticks(ticks=[1, 2, 3, 4], labels=custom_labels, fontsize=20)
plt.yticks(fontsize=20)

# Set background color to white
plt.gca().set_facecolor('white')

# Set spines color to gray
plt.gca().spines['top'].set_color('#888888')
plt.gca().spines['right'].set_color('#888888')
plt.gca().spines['bottom'].set_color('#888888')
plt.gca().spines['left'].set_color('#888888')

# Show the plot
plt.tight_layout()
plt.show()
'''
####################################################################
# t statistic analysis
####################################################################
'''
# t statistic for PCr/ATP

# Split the data based on groups in column B
group1 = df[df['Group'] == 'Control']['PCr_ATP_Baseline']
group2 = df[df['Group'] == 'Intervention']['PCr_ATP_Baseline']

# Calculate the t-statistic and p-value
t_stat, p_value = ttest_ind(group1, group2)
# Calculate Cohen's d
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

# Print the result
print(f"T-statistic for PCr/ATP: {t_stat}")
print(f"P-value for PCr/ATP: {p_value}")
print(f"Cohen's d for PCr/ATP: {cohens_d}")

# t statistic for GAD7

# Split the data based on groups in column B
group1 = df[df['Group'] == 'Control']['GAD7_Baseline']
group2 = df[df['Group'] == 'Intervention']['GAD7_Baseline']

# Calculate the t-statistic and p-value
t_stat, p_value = ttest_ind(group1, group2)
# Calculate Cohen's d
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

# Print the result
print(f"T-statistic for GAD7: {t_stat}")
print(f"P-value for GAD7: {p_value}")
print(f"Cohen's d for GAD7: {cohens_d}")

# t statistic for waist

# Split the data based on groups in column B
group1 = df[df['Group'] == 'Control']['Waist_Circumference_Baseline']
group2 = df[df['Group'] == 'Intervention']['Waist_Circumference_Baseline']

# Calculate the t-statistic and p-value
t_stat, p_value = ttest_ind(group1, group2)
# Calculate Cohen's d
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

# Print the result
print(f"T-statistic for waist: {t_stat}")
print(f"P-value for waist: {p_value}")
print(f"Cohen's d for waist: {cohens_d}")

# t statistic for age

# Split the data based on groups in column B
group1 = df[df['Group'] == 'Control']['Age']
group2 = df[df['Group'] == 'Intervention']['Age']

# Calculate the t-statistic and p-value
t_stat, p_value = ttest_ind(group1, group2)
# Calculate Cohen's d
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

# Print the result
print(f"T-statistic for age: {t_stat}")
print(f"P-value for age: {p_value}")
print(f"Cohen's d for age: {cohens_d}")

# t statistic for PCr/ATP at 15 months

# Split the data based on groups in column B
group1 = df[df['Group'] == 'Control']['PCr_ATP_Week_15']
group2 = df[df['Group'] == 'Intervention']['PCr_ATP_Week_15']

# Calculate the t-statistic and p-value
t_stat, p_value = ttest_ind(group1, group2)
# Calculate Cohen's d
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

# Print the result
print(f"T-statistic for PCr/ATP outcome: {t_stat}")
print(f"P-value for PCr/ATP outcome: {p_value}")
print(f"Cohen's d for PCr/ATP outcome: {cohens_d}")
'''
####################################################################
# ANOVA analysis
####################################################################

# Define the model using ordinary least squares (OLS)
model = ols('PCr_ATP_Difference ~ C(Group)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Print the ANOVA table
print(f"anova table for PCr-ATP: {anova_table}")

# Define the model using ordinary least squares (OLS)
model = ols('GAD7_Difference ~ C(Group)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Print the ANOVA table
print(f"anova table for GAD7: {anova_table}")

# Perform Tukey post hoc test
tukey = pairwise_tukeyhsd(endog=df['GAD7_Difference'], groups=df['Group'], alpha=0.01)
print(f"tukey table for GAD7: {tukey}")

# Function to calculate Cohen's d
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_std
    return d, pooled_std

# Function to calculate Hedges' g
def hedges_g(d, n1, n2):
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * correction
    return g

# Function to calculate the 95% confidence interval for Cohen's d and Hedges' g
def effect_size_ci(effect_size, n1, n2, pooled_std, alpha=0.05):
    se = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))
    z = 1.96  # 95% CI
    lower = effect_size - z * se
    upper = effect_size + z * se
    return lower, upper

# Calculate Cohen's d, Hedges' g, and confidence intervals for each pairwise comparison
groups = df['Group'].unique()
effect_size_results = []

for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        group1 = df[df['Group'] == groups[i]]['GAD7_Difference']
        group2 = df[df['Group'] == groups[j]]['GAD7_Difference']
        d, pooled_std = cohen_d(group1, group2)
        g = hedges_g(d, len(group1), len(group2))
        d_ci_lower, d_ci_upper = effect_size_ci(d, len(group1), len(group2), pooled_std)
        g_ci_lower, g_ci_upper = effect_size_ci(g, len(group1), len(group2), pooled_std)
        effect_size_results.append((groups[i], groups[j], d, d_ci_lower, d_ci_upper, g, g_ci_lower, g_ci_upper))

# Display results
print("\nEffect Size Results with 95% Confidence Intervals:")
for res in effect_size_results:
    print(f"{res[0]} vs {res[1]}: Cohen's d = {res[2]:.3f}, 95% CI = ({res[3]:.3f}, {res[4]:.3f}), "
          f"Hedges' g = {res[5]:.3f}, 95% CI = ({res[6]:.3f}, {res[7]:.3f})")


# Calculate Eta-squared
anova_table['sum_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
anova_table = anova_table.rename(columns={'sum_sq': 'eta_sq'})
print(anova_table)

####################################################################
# Regression analysis
####################################################################
'''
# Prepare the data
# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['Group'], drop_first=True)

X = df[['Group_Intervention', 'Age', 'Waist_Circumference_Baseline', 'GAD7_Baseline', 'PCr_ATP_Baseline']]  # Independent variables

y = df['GAD7_Difference']  # Dependent variable

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

print(model.summary())
'''
####################################################################
# Data extract
####################################################################
'''
# Extract data to .csv file
df.to_csv('synoptic_data.csv', index=False)
'''