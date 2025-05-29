import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2
from iminuit.cost import LeastSquares

def scarica(t,V_0, tau):
    return V_0*np.exp(-t / (tau))
def carica(t,V_0,tau):
    return V_0*(1 - np.exp(-t / (tau)))


Resistance = 21.78   #  KOhm
Capacitance = 97    # nanoFarad
err_resistance = 0.9/100 * Resistance  + 0.01 # KOhm, dal modello FLUKE 111
print(err_resistance)
err_capacitance = 1.9/100 * Capacitance + 1   # nFarad
print(err_capacitance)
tempi = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.72, 2.04, 2.36, 2.64, 3.00, 3.40, 3.8, 4.2, 4.6, 5, 5.4, 5.80, 6.2, 6.6, 7, 8, 9])
voltaggi = np.array([3.9, 3.58, 3.26, 2.98, 2.74, 2.46, 2.22, 2.04, 1.74, 1.50, 1.28, 1.12, 0.96, 0.8, 0.68, 0.56, 0.46, 0.38, 0.32, 0.26, 0.22, 0.2, 0.18, 0.1, 0.06])
err_voltaggi = [
0.02,  
0.02,  
0.02,  
0.02,  
0.02,  
0.02,  
0.02,  
0.01,  
0.02,  
0.02,  
0.01,  
0.01,  
0.01, 
0.01, 
0.01, 
0.01, 
0.02, 
0.02, 
0.01, 
0.02,
0.02,
0.01,
0.02,
0.02,
0.02
]
voltage_costante = 3.90 
err_voltage_costante = 0.02
#carica :


voltage_carica = voltage_costante*np.ones(len(tempi))-np.array(voltaggi)
print(voltage_carica)
err_voltage_carica = [0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02, 
0.02, 0.02, 0.01, 0.02, 0.02, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01]

#interpolazione per la carica:
my_cost_func_carica = LeastSquares(tempi, voltage_carica, err_voltage_carica, carica)
m_carica = Minuit(my_cost_func_carica,V_0 = 3.90, tau= 2)
m_carica.migrad()
# Estrazione risultati per la carica
tau_fit_carica = m_carica.values['tau']
V_fit_carica = m_carica.values['V_0']
err_V_fit_carica = m_carica.errors['V_0']
tau_err_carica = m_carica.errors['tau']
chi2_val_carica = m_carica.fval
chi2_red_carica = chi2_val_carica / m_carica.ndof
print(m_carica.ndof)# un solo parametro
print(len(tempi)-1)
#graficarizzazione per la carica
plt.figure(figsize=(10, 6))
plt.errorbar(tempi, voltage_carica, yerr=err_voltage_carica, fmt='o', label='Dati Carica')
plt.plot(tempi, carica(tempi,V_fit_carica, tau_fit_carica), 'r-', label=f'V_0 = {V_fit_carica:.3f}+- {err_V_fit_carica:.3f}\n τ= {tau_fit_carica:.3f} ± {tau_err_carica:.3f}')
plt.xlabel('tempo (ms)')
plt.ylabel('voltaggio (V)')
plt.title('circuito RC: Fit Carica')
plt.grid(True)
plt.legend()
# Annotazioni
textstr_carica = f"$\chi^2_{{rid}}$ = {chi2_red_carica:.2f}\n$p$-value = {1 - chi2.cdf(chi2_val_carica, m_carica.ndof):.3f}"
plt.text(0.05, max(voltage_carica)*0.7, textstr_carica, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()
#estrapolazione capacità per la carica:
R_carica = Resistance
C_carica = (tau_fit_carica/R_carica)*10**3   # in nF
print(f"Capacitance Carica: {C_carica:.2f} nF")
err_C_carica = np.sqrt((tau_err_carica/R_carica)**2 + (tau_fit_carica*err_resistance/(R_carica**2))**2)*10**3
print(f"Errore Capacitance Carica: {err_C_carica:.2f} nF")
#compatibilità con il valore atteso:
n = abs(C_carica - Capacitance) / np.sqrt(err_C_carica**2 + err_capacitance**2)
print(f"Compatibilità con il valore atteso: {n:.2f} σ")



#scarica:
voltage_scarica = [3.90, 
3.58-0.01,  
3.26 +0.01,  
2.98 -0.02,  
2.74 +0.01, 
2.46 +0.01,
2.22 -0.01,
2.04 +0.01,
1.74 -0.01,
1.50 +0.02,
1.28 -0.01,
1.12 +0.01,
0.96 +0.01,
0.80 -0.01,
0.68 +0.01,
0.56 -0.01,
0.46 +0.01,
0.38 -0.01,
0.32 +0.01,
0.26 -0.01,
0.22 +0.01,
0.20 -0.01,
0.18 +0.01,
0.10 +0.01,
0.06
]


my_cost_func = LeastSquares(tempi, voltage_scarica, err_voltaggi, scarica)
m = Minuit(my_cost_func, V_0 = 3.90,tau= 2)
m.migrad()

# Estrazione risultati
tau_fit = m.values['tau']
V_fit = m.values['V_0']
err_V_fit = m.errors['V_0']
tau_err = m.errors['tau']
chi2_val = m.fval
ndof = len(tempi) - 2 # un solo parametro
chi2_red = chi2_val / ndof
p_value = 1 - chi2.cdf(chi2_val, ndof)


plt.figure(figsize=(10, 6))
plt.errorbar(tempi, voltage_scarica, yerr=err_voltaggi, fmt='o', label='Dati')
plt.plot(tempi, scarica(tempi,V_fit,tau_fit), 'r-', label=f'V_0 = {V_fit:.2f}+-{err_V_fit:.2f}\n τ= {tau_fit:.3f} ± {tau_err:.3f}')
plt.xlabel('tempo (ms)')
plt.ylabel('voltaggio (V)')
plt.title('circuito RC: Fit Scarica')
plt.grid(True)
plt.legend()

# Annotazioni
textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
plt.text(7.50, max(voltaggi)*0.7, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
C = (tau_fit/Resistance)*10**3
err_C = np.sqrt((tau_err/Resistance)**2 + (tau_fit*err_resistance/(Resistance**2))**2)*10**3
print(f"Capacitance_scarica: {C:.2f} nF")
print(f"Errore Capacitance_scarica: {err_C:.2f} nF")
# Compatibilità con il valore atteso
n = abs(C - Capacitance) / np.sqrt(err_C**2 + err_capacitance**2)
print(f"Compatibilità con il valore atteso: {n:.2f} σ")




