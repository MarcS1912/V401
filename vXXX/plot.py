import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.stats import poisson
import numpy as np

def Data(Name):
    Daten1 = pd.read_csv(Name, skiprows=0, sep=";")
    Daten2 = Daten1.replace(",", ".", regex=True)
    return Daten2.to_numpy(dtype=np.float64)

def max_exponent_2d(array):
    max_exp = []
    for i in range(len(array[0])):
        max_exp.append(int(np.floor(np.log10(np.abs(array[:,i].max())))))
    return max_exp

def min_exponent_2d(array):
    min_exp = []
    for i in range(len(array[0])):
        non_zero_values = array[:,i][np.nonzero(array[:,i])]
        if len(non_zero_values) == 0:
            min_exp.append(0)
        else:
            min_exp.append(int(np.floor(np.log10(np.abs(abs(non_zero_values).min())))))
    return min_exp

def array_to_latex_table(array, filename):
    exponent=max_exponent_2d(array)
    minexponent=min_exponent_2d(array)
    with open(filename, "w") as f:
        for row in array:
            formatted_row = []
            i=0
            for cell in row:
                if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <=5 :
                    formatted_row.append("{:.0f}".format(cell))
                elif exponent[i] < -2:
                    formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                elif exponent[i] >5:
                    formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                elif (10*cell).is_integer():
                    formatted_row.append("{:.1f}".format(cell).replace(".", ","))
                else:
                    formatted_row.append("{:.2f}".format(cell).replace(".", ","))
                i=i+1
            f.write(" & ".join(formatted_row))
            f.write(" \\\\\n")
    return minexponent
def array_to_latex_table_1D(array, filename):
    exponent=max_exponent_2d(array)
    minexponent=min_exponent_2d(array)
    with open(filename, "w") as f:
        formatted_array = []
        i=0
        for cell in array:
            formatted_array=[]
            if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <= 5:
                formatted_array.append("{:.0f}".format(cell))
            elif exponent[i] < -2:
                formatted_array.append("{:.2f}".format(cell * 10**-minexponent[i], -minexponent[i]).replace(".", ","))
            elif exponent[i] >= 5:
                formatted_array.append("{:.2f}".format(cell * 10**-minexponent[i], -minexponent[i]).replace(".", ","))
            else:
                formatted_array.append("{:.2f}".format(cell).replace(".", ","))
           
            f.write(", ".join(formatted_array))
            f.write(" \\\\\n")
    return minexponent
def deff(d,ü):
    return 2*d/ü
def Delta_deff(Delta_d,ü):
    return 2*Delta_d/ü
def lamda(Deff,z):
    return 2*Deff/(2*z+1)
def dlamda(Lamda,ddeff,Deff,z,dz):
    return Lamda*np.sqrt((ddeff/Deff)**2+(2*dz/(2*z+1))**2)
def Wellenlaenge(Laenge,Delta_d,ü):
    d=np.mean(Laenge[:,0])*10**-3
    z=np.mean(Laenge[:,1])
    dz=np.std(Laenge[:,1],ddof=1)/np.sqrt(len(Laenge[:,1]))
    print(f"z={z}±{dz}")
    Deff=deff(d,ü)
    delta_deff=Delta_deff(Delta_d,ü)
    print(f"Weglänge des Lichts:{Deff}±{delta_deff}")
    laenge=lamda(Deff,z)
    dlaenge=dlamda(laenge,delta_deff,Deff,z,dz)
    return np.array([laenge,dlaenge])

def gMittelwert(lamb1,lamb2):
    p1=1/lamb1[1]**2
    p2=1/lamb2[1]**2
    lamb=(p1*lamb1[0]+p2*lamb2[0])/(p1+p2)
    dlamb=1/np.sqrt(p1+p2)
    return np.array([lamb,dlamb])
def Abweichung(exp,theo):
    return abs(exp-theo)/theo

def dn(z,lamda,b):
    return (z*lamda)/(2*b)
def Ddn(n,dz,z):
    return n*dz/z
def n(dn,t,t0,p0,dp):
    return 1+dn*t*p0/(t0*dp)
def Dn(n,Ddn,dn,Ddp,dp,dt,t):
    return (n-1)*np.sqrt((Ddn/dn)**2+(Ddp/dp)**2+(dt/t)**2)
def meanz(z):
    return np.array([np.mean(z),np.std(z,ddof=1)/np.sqrt(len(z))])

def Index(z,t0,t,p0,p,dp,lamda,b,DT):
    mz=meanz(z)
    print(mz)
    dN=dn(mz[0],lamda,b)
    DdN=Ddn(dN,mz[1],mz[0])
    print(f"Änderung von n: {dN}±{DdN}")
    N=n(dN,t,t0,p0,p)
    DN=Dn(N,DdN,dN,dp,p,DT,t)
    return np.array([N,DN])

Index_Minima=Data("content/Index.CSV")
Laenge=Data("content/Laenge.CSV")
exp2=array_to_latex_table(Index_Minima,"build/Tabelle2.tex")
exp1=array_to_latex_table(Laenge,"build/Tabelle1.tex")
Delta_d=0.01e-3
ü=5.017
b=50e-3

lamb1=Wellenlaenge(Laenge[2:],Delta_d,ü)
print(f"Wellenlänge für d=5mm: {lamb1[0]:.3e}± {lamb1[1]:.3e}")
lamb2=Wellenlaenge(Laenge[:3],Delta_d,ü)
print(f"Wellenlänge für d=5.1mm: {lamb2[0]:.3e}± {lamb2[1]:.3e}")
lamb=gMittelwert(lamb1,lamb2)
print(f"gemittelte Wellenlänge: {lamb[0]:.3e}± {lamb[1]:.3e}")
theo=680e-9
print(f"relativer Fehler liegt bei: {Abweichung(lamb[0],theo):.3f}")

T0=273.15
p0=1013.2
T=297.15
DT=2
p=666.6
dp=26.66

Index_raus=Index(Index_Minima[::2],T0,T,p0,p,dp,theo,b,DT)
Index_rein=Index(Index_Minima[1::2],T0,T,p0,p,dp,theo,b,DT)
print(f"Brechungsindex in Luft,bei Evakuierung:{Index_raus[0]:.6e}± {Index_raus[1]:.6e}")
print(f"Brechungsindex in Luft,beim Lufteinlass:{Index_rein[0]:.6e}± {Index_rein[1]:.6e}")
ntheo=1.0002911
print(f"relativer Fehler1 liegt bei: {Abweichung(Index_raus[0],ntheo):.6f}")
print(f"relativer Fehler2 liegt bei: {Abweichung(Index_rein[0],ntheo):.6f}")
print("Aussagekräftige relative Abweichung")
print(f"relativer Fehler1 liegt bei: {Abweichung(Index_raus[0]-1,ntheo-1):.6f}")
print(f"relativer Fehler2 liegt bei: {Abweichung(Index_rein[0]-1,ntheo-1):.6f}")

ngemein=gMittelwert(Index_rein,Index_raus)
print(f"Brechungsindex in Luft,unter ein Beziehung aller Messungen:{ngemein[0]:.6e}± {ngemein[1]:.6e}")
print(f"relativer Fehler gemeinsam liegt bei: {Abweichung(ngemein[0],ntheo):.6f}")
print(f"relativer Fehler gemeinsam aussagekräftig liegt bei: {Abweichung(ngemein[0]-1,ntheo-1):.6f}")