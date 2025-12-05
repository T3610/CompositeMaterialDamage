import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_excel('DamageData.xlsx')
print(df.head())

df2 = df.loc[df["Has Preexisting Damage"]] 
input(df2["Has Preexisting Damage"])
x = df2["Damage Width"]
y1 = df2["Horizontal Arm Length"]
y2 = df2["Vert Arm Width"]
yc = (y1+y2)*2
plt.title("Width of Damage area vs Arm Dimensions")
plt.scatter(x,y1, label="Horizontal Length",color="blue")
plt.scatter(x,y2, label="Vertical Width",color="orange")
plt.scatter(x,yc, label="Corrected",color="green")
plt.legend()
plt.axline([0,0],[1,1])

hasDamageDF = df.loc[df["Has Preexisting Damage"]=="False"]
x = df["Damage Width"]
y1 = df["Horizontal Arm Length"]
y2 = df["Vert Arm Width"]
yc = (y1+y2)*2
plt.scatter(x,y1, label="Horizontal Length",marker="*", color="blue")
plt.scatter(x,y2, label="Vertical Width", marker="*",color="orange")
plt.scatter(x,yc, label="Corrected", marker="*",color="green")

plt.show()



x = df["Damage Height"]
y1 = df["Horizontal Arm Width"]
y2 = df["Vert Arm Length"]
yc = (y1+y2)*2
plt.title("Height of Damage area vs Arm Dimensions")
plt.scatter(x,y1, label="Horizontal Width")
plt.scatter(x,y2, label="Vertical Length")
plt.scatter(x,yc, label="Corrected")
plt.legend()
plt.axline([0,0],[1,1])
plt.show()
