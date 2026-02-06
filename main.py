import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import itertools

import DamageMorphology as DM
import graphAnalysis as GA
import MergeGraphAndImageData as MD

def plotAllVsAll():
    

    # Load the data
    df = pd.read_csv('ALLDATA.csv')
    

    # 1. Load and Filter Data
    df = pd.read_csv('ALLDATA.csv')
    df['Group'] = df['ID'].str[:2]
    #df_filtered = df[df['Group'] != 'TB'].copy()
    df_filtered = df.copy()
    # 2. Define Plotting Function
    def plot_with_equations(df, x_col, y_col):
        plt.figure(figsize=(12, 8))
        
        # Scatter points with symbol for group and color for damage
        sns.scatterplot(
            data=df, x=x_col, y=y_col, 
            hue='Has Preexisting Damage', palette={True: 'red', False: 'blue'},
            style='Group', s=150, alpha=0.8, edgecolor='k'
        )
        
        # 3. Calculate and Plot Regression Lines
        for status, color in zip([True, False], ['red', 'blue']):
            subset = df[df['Has Preexisting Damage'] == status][[x_col, y_col]].dropna()
            if len(subset) > 1:
                slope, intercept, r_val, p_val, _ = stats.linregress(subset[x_col], subset[y_col])
                
                x_fit = np.array([subset[x_col].min(), subset[x_col].max()])
                y_fit = slope * x_fit + intercept
                
                # Formulate Equation String
                label_str = f'Fit ({status}): $y = {slope:.3f}x + {intercept:.3f}$ ($R^2={r_val**2:.2f}$)'
                plt.plot(x_fit, y_fit, color=color, linestyle='--', alpha=0.7, label=label_str)

        # 4. Add ID Labels
        for i in df.index:
            if pd.notnull(df.loc[i, x_col]) and pd.notnull(df.loc[i, y_col]):
                plt.text(df.loc[i, x_col], df.loc[i, y_col], df.loc[i, 'ID'], fontsize=9)

        plt.title(f'{y_col} vs {x_col} with Trend Equations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'ScatterPlots/{y_col}_vs_{x_col}_with_equations.png', dpi=100)
        #plt.show()

    # Example usage:

    numeric_columns = [
        'reboundEnergy', 'absorbedEnergy', 'maxForce', 
        'Total Area of Damage (mm^2)', 'Damage Width', 'Damage Height',
        'Vert Arm Length', 'Vert Arm Width', 'Horizontal Arm Length', 
        'Horizontal Arm Width', 'Distance from Impactor to Damage Center (mm)'
    ]
    column_pairs = list(itertools.combinations(numeric_columns, 2))
    for x_col, y_col in column_pairs:
        plot_with_equations(df_filtered, x_col, y_col)


runDM = True
runGA = True
if runDM:
    DM.main()
if runGA:
    GA.main()

 # merge the data
MD.merge_data() # saves to ALLDATA.xlsx
if False:
    plotAllVsAll()

df = pd.read_csv('ALLDATA.csv')
dfDropped = df.drop(columns=['Unnamed: 0','ID','Has Preexisting Damage'])
corr_matrix = dfDropped.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Correlation Matrix with Scores')
plt.tight_layout()
plt.show()
