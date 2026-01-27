import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import cumulative_trapezoid

ImageDF = pd.read_csv("DamageData.csv")
ImpactDF = pd.read_csv("EnergyForceData.csv")
mergedDF = pd.merge(ImpactDF, ImageDF, on='ID', how='outer')

print(mergedDF)
mergedDF.to_excel("ALLDATA.xlsx")