import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import cumulative_trapezoid
# --- Step 1: Load CSV ---
# Assume CSV has columns: 'time', 'force'
data = pd.read_csv("ImpactData/TB1.csv",skiprows=9500,delimiter=" ") #skiprows=9953

data.columns = ["Point ID", "Time [ms]", "Force [N]", "Voltage Ch1 [mV]"]

time = data["Time [ms]"]
force = data["Force [N]"]

mass = 4.125

duration = 80*(200)#seconds
dur2 = time[duration]-time[0]

print("duration",duration)
print("duration2",dur2)
meanForce = 43
time = time[:int(duration)]

force = force[:int(duration)]+meanForce



ENERGY = 30
g = 9.81
initalVel = np.sqrt(30/(0.5*mass))
initalHeight = ENERGY/(mass*g)
accel = force/mass

accel = accel.to_numpy()
time = time.to_numpy()
time = time.reshape(-1)
time = time/1000
#vt = initalVel-((force/mass)*(0.000005))
vt =  cumulative_trapezoid(accel, time, initial=0)
vt = initalVel - vt
defl = cumulative_trapezoid(vt, time, initial=0)
defl = defl

ea = cumulative_trapezoid(force, defl, initial=0)
#print(vt)
plt.subplot(1,4,1)
plt.plot(time,force)
plt.title("force time")

plt.subplot(1,4,2)
plt.plot(time,vt)
plt.title("actuall velocity time")

plt.subplot(1,4,3)
plt.plot(time,defl)
plt.title("displacement time")

plt.subplot(1,4,4)
plt.plot(time,ea)
plt.title("energy time")

plt.show()