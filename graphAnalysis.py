import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import cumulative_trapezoid
def main():
    setNames = ["MA","TB","ZA","AS"]
    IDnumber = ["1","2","3","4","5"]
    duration = 100*(200)#seconds
    g = 9.81
    dataList = []
    for set in setNames:
        if set == "TB":
            ENERGY = 30
            duration = 100*(200)#seconds
            mass = 4.125
            initalVel = np.sqrt(ENERGY/(0.5*mass))
            initalHeight = ENERGY/(mass*g)

        else:
            ENERGY = 50
            duration = 150*(200)#seconds
            mass = 5.125
            initalVel = np.sqrt(ENERGY/(0.5*mass))
            initalHeight = ENERGY/(mass*g)

        for id in IDnumber:
            
            string = set+id
            if string != "ZA1":
                    
                data = pd.read_csv("ImpactData/" + string + ".csv",skiprows=9500,delimiter=" ") #skiprows=9953

                data.columns = ["Point ID", "Time [ms]", "Force [N]", "Voltage Ch1 [mV]"]

                time = data["Time [ms]"]
                force = data["Force [N]"]
                #dur2 = time[duration]-time[0]
                meanForce = 43
                time = time[:int(duration)]
                force = force[:int(duration)]+meanForce
                print(type(force))
                force = pd.Series(np.convolve(force, np.ones(10)/10, mode='same'))
                
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

                maxForce = np.max(force)
                maxForceTime = np.argmax(force)
                maxForceTime = time[maxForceTime]
                

                maxEnergy = np.max(ea)
                maxEnergyTime = np.argmax(ea)
                maxEnergyTime = time[maxEnergyTime]
                endEnergy = ea[-1]
                reboundEnergy = maxEnergy-endEnergy
                absorbedEnergy = endEnergy
                impactEnergy = maxEnergy
                
                data = [string,reboundEnergy,absorbedEnergy,impactEnergy,maxForce]
                dataList.append(data)
                
                fig, axs = plt.subplots(3, 1)
                time = time #convert to milliseconds

                fig.suptitle('Impact Analysis for ' + string)
                axs[0].plot(time,force)
                axs[0].set_xlabel('Time (s)')
                axs[0].set_ylabel("Force (N)")
                axs[0].axvline(maxForceTime,label = "max force", linestyle = "--")

                axs[1].plot(time, vt, 'tab:orange')
                axs[1].set_xlabel('Time (s)')
                axs[1].set_ylabel("Velocity (m/s)")

                axs[2].plot(time,defl, 'tab:green')
                axs[2].set_xlabel('Time (s)')
                axs[2].set_ylabel("Displacement (m)")
                plt.tight_layout()
                plt.savefig("InterrimReport/ImpactPlots/ForceVelocityDisplacement/"+string+".jpeg",dpi=300)
                plt.close()

                
                fig, axs = plt.subplots(2, 1)

                axs[0].plot(time, ea, 'tab:red')
                axs[0].set_xlabel('Time (s)')
                axs[0].set_ylabel("Energy (J)")
                #axs[0].axvline(maxEnergyTime, linestyle = "--")

                axs[0].axhline(endEnergy, linestyle = "--", label = "Energy Rebounded")

                axs[0].axhline(maxEnergy,linestyle = "--", label = "Max Energy")

                axs[1].plot(defl, force, 'tab:red')
                axs[1].set_xlabel('Deflection (m)')
                axs[1].set_ylabel("Force (N)")

                """axs[2].plot(time, ea, 'tab:red')
                axs[2].set_xlabel('Time (s)')
                axs[2].set_ylabel("Energy (J)")"""
                plt.tight_layout()
                plt.savefig("InterrimReport/ImpactPlots/EnergyDeflection/"+string+".jpeg",dpi=300)
                plt.close()

                

    df = pd.DataFrame(dataList,columns=["ID","reboundEnergy","absorbedEnergy","impactEnergy","maxForce"])
    df.to_csv("EnergyForceData.csv", index=False)
    print("Graph Analysis Complete")
    