from turtle import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

distanceDF = pd.read_csv("distance.csv")
energyDF = pd.read_csv("EnergyForceData.csv")
photoDF = pd.read_csv("DamageData.csv")
mergedDF = pd.merge(energyDF, distanceDF, on='ID', how='outer')
mergedDF = pd.merge(mergedDF, photoDF, on='ID', how='inner')
#damageData = pd.read_csv("DamageData.csv")
#mergedDF = pd.merge(mergedDF, damageData, on='ID', how='outer')
print(mergedDF)
mergedDF.to_excel("ALLDATA.xlsx")

def CreateEnergyDistancePlot(DF):
    
    #print(DF.columns)
    holedDF = DF.loc[DF["Distance"]>0]
    #print(holedDF.columns)
    nonHoledDF = DF.loc[DF["Distance"] == 0]
    print(nonHoledDF)
    energy = holedDF["absorbedEnergy"]
    distance = holedDF["Distance"]
    energy = energy.tolist()
    distance = distance.tolist()
    #print(energy)
    #print(distance)
    colors = np.where(holedDF["Has Preexisting Damage"], 'orange', 'purple')
    plt.scatter(energy,distance,c=colors)
    plt.xlabel("Absorbed Energy (J)")
    plt.ylabel("Distance from center of hole to center of impact")
    plt.axvline(nonHoledDF["absorbedEnergy"][0],ls = "--")

    labels = holedDF["ID"].tolist()
    for i, txt in enumerate(labels):
        plt.annotate(txt, (energy[i], distance[i]))
    plt.show()

def DamageAreaVsDistance(DF):
    holedDF = DF.loc[DF["Distance"]>0]
    #print(holedDF.columns)
    nonHoledDF = DF.loc[DF["Distance"] == 0]
    damageArea = holedDF["Total Area of Damage (mm^2)"].tolist()
    distance = holedDF["Distance"].tolist()
    labels = holedDF["ID"].tolist()
    print(labels)

    colors = np.where(holedDF["Has Preexisting Damage"], 'orange', 'purple')
    plt.scatter(distance,damageArea,c=colors)
    plt.ylim(0,600)

    plt.xlabel("Distance (mm)")
    plt.ylabel("Damaged Area (mm^2)")
    for i, txt in enumerate(labels):
        plt.annotate(txt, (distance[i],damageArea[i]))

    
    plt.show()

def CreateImpactWidthDistancePlot(DF):
    
    #print(DF.columns)
    holedDF = DF.loc[DF["Distance"]>0]
    #print(holedDF.columns)
    nonHoledDF = DF.loc[DF["Distance"] == 0]
    print(nonHoledDF)
    width = holedDF["Damage Width"].tolist()
    distance = holedDF["Distance"].tolist()
    
    #print(energy)
    #print(distance)
    colors = np.where(holedDF["Has Preexisting Damage"], 'orange', 'purple')
    plt.scatter(distance,width,c=colors)
    plt.xlabel("Distance from impact to hole center(mm)")
    plt.ylabel("Area Width (mm)")
    #plt.axvline(nonHoledDF["absorbedEnergy"][0],ls = "--")

    labels = holedDF["ID"].tolist()
    for i, txt in enumerate(labels):
        plt.annotate(txt, (distance[i],width[i]))

    plt.ylim(0,800)
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# df = your DataFrame

# Select only numeric columns
df = mergedDF
numeric_cols = df.select_dtypes(include='number').columns

for i, col_x in enumerate(numeric_cols):
    for j, col_y in enumerate(numeric_cols):
        if i >= j:
            continue  # avoid duplicates and self-plots

        plt.figure(figsize=(6, 4))
        plt.scatter(df[col_x], df[col_y], alpha=0.7)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"{col_x} vs {col_y}")
        plt.tight_layout()
        labels = df["ID"].tolist()
        for i, txt in enumerate(labels):
            plt.annotate(txt, (df[col_x][i],df[col_y][i]))
        plt.show()


"""CreateImpactWidthDistancePlot(mergedDF)
DamageAreaVsDistance(mergedDF)
CreateEnergyDistancePlot(mergedDF)
#CreateEnergyDistancePlot(mergedDF)"""