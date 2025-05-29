import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2
from iminuit.cost import LeastSquares

def scarica(t,V_0_scarica, tau_scarica):
    return V_0_scarica*(1-np.exp(-t / (tau_scarica)))

def carica(t,V_0, tau):
    return V_0*np.exp(-t / (tau))


Resistance = 0.995 # KOhm
err_resistance = 0.9/100 * Resistance + 0.001 # KOhm, dal modello FLUKE 111
print(err_resistance)
tempi = np.array([6, 14, 22, 30, 38, 46, 56, 62, 70, 78, 86, 94, 102, 118, 130, 156, 180, 206]) # # in microsecondi
voltaggi = 1.80*np.ones(len(tempi)) + np.array([-1.40, -0.880, -0.440, -0.08, 0.24, 0.48, 0.72, 0.92, 1.08, 1.20, 1.28, 1.36, 1.44, 1.56, 1.64, 1.72, 1.80, 1.84]) #in volt
err_voltaggi = np.array([np.sqrt(2)*0.04, np.sqrt(5)*0.04, np.sqrt(2)*0.04, np.sqrt(5)*0.04, np.sqrt(5)*0.04, np.sqrt(5)*0.04, np.sqrt(5)*0.04, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(5)*0.04, np.sqrt(5)*0.04, np.sqrt(5)*0.04, np.sqrt(5)*0.04, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(5)*0.04]) #in Volt
#carica:
voltage_costante = 1.80+1.84
voltage_carica = voltage_costante - voltaggi
print(voltage_carica)
err_voltage_carica = np.array([
np.sqrt(5)*0.04,
np.sqrt(2)*0.04,
np.sqrt(5)*0.04,
np.sqrt(5)*0.04,
np.sqrt(2)*0.04,
np.sqrt(2)*0.04,
np.sqrt(2)*0.04,
np.sqrt(5)*0.04,
np.sqrt(5)*0.04,
np.sqrt(5)*0.04,
np.sqrt(5)*0.04,
np.sqrt(2)*0.04,
np.sqrt(5)*0.04,
np.sqrt(5)*0.04,
np.sqrt(2)*0.04,
np.sqrt(2)*0.04,
np.sqrt(2)*0.04,
np.sqrt(5)*0.04
]
)
print(err_voltage_carica)
#interpolazione per la carica:
my_cost_func_carica = LeastSquares(tempi, voltage_carica, err_voltage_carica, carica)
m_carica = Minuit(my_cost_func_carica,V_0 = 3.64, tau= 50)
m_carica.migrad()
# Estrazione risultati per la carica

tau_fit_carica = m_carica.values['tau']
V_fit_carica = m_carica.values['V_0']
err_V_fit_carica = m_carica.errors['V_0']
tau_fit_carica = m_carica.values['tau']
tau_err_carica = m_carica.errors['tau']
chi2_val_carica = m_carica.fval
ndof_carica = len(tempi) - 2  # un solo parametro
chi2_red_carica = chi2_val_carica / ndof_carica
p_value_carica = 1 - chi2.cdf(chi2_val_carica, ndof_carica)
print(f"Carica: V_0 = {V_fit_carica:.2f}+-{err_V_fit_carica:.2f}\n tau = {tau_fit_carica:.3f} ± {tau_err_carica:.3f}, chi2_red = {chi2_red_carica:.2f}, p-value = {p_value_carica:.3f}")
#grafico carica:
plt.figure(figsize=(10, 6))
plt.errorbar(tempi, voltage_carica, yerr=err_voltage_carica, fmt='o', label='Dati Carica')
plt.plot(tempi, carica(tempi,V_fit_carica, tau_fit_carica), 'r-', label=f'V_0 = {V_fit_carica:.2f}+-{err_V_fit_carica:.2f}\n  τ = {tau_fit_carica:.3f} ± {tau_err_carica:.3f}')
plt.xlabel('tempo (μs)')
plt.ylabel('voltaggio (V)')
plt.title('circuito RL: Fit della Carica')
plt.grid(True)
plt.legend()
# Annotazioni
textstr_carica = f"$\chi^2_{{rid}}$ = {chi2_red_carica:.2f}\n$p$-value = {p_value_carica:.3f}"
plt.text(180.50, max(voltage_carica)*0.7, textstr_carica, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

#estrapolazione induttanza:
inductance = tau_fit_carica * Resistance   # in milli Henry
err_inductance = np.sqrt((tau_err_carica * Resistance)**2 + (tau_fit_carica * err_resistance)**2)  # in milli Henry
print(f"Induttanza Carica: {inductance:.2f} ± {err_inductance:.2f} mH")


#scarica:

voltage_scarica = 1.80*np.ones(len(tempi)) + np.array([-1.40, -0.87, -0.45, -0.09, 0.25, 0.47, 0.73, 0.91, 1.09, 1.19, 1.29, 1.35, 1.45, 1.55, 1.65, 1.71, 1.80, 1.84])
print(voltage_scarica)
err_voltage_scarica = err_voltaggi
print(err_voltage_scarica)
#interpolazione per la scarica:
my_cost_func = LeastSquares(tempi, voltage_scarica, err_voltage_scarica, scarica)
m = Minuit(my_cost_func,V_0_scarica = 3.64, tau_scarica= 50)
m.migrad()

# Estrazione risultati

tau_fit = m.values['tau_scarica']
V_fit = m.values['V_0_scarica']
err_V_fit = m.errors['V_0_scarica']
tau_err = m.errors['tau_scarica']
chi2_val = m.fval
ndof = len(tempi) - 2  # un solo parametro
chi2_red = chi2_val / ndof
p_value = 1 - chi2.cdf(chi2_val, ndof)


plt.figure(figsize=(10, 6))
plt.errorbar(tempi, voltaggi, yerr=err_voltaggi, fmt='o', label='Dati')
plt.plot(tempi, scarica(tempi,V_fit,tau_fit), 'r-', label=f'V_0 = {V_fit:.2f}+-{err_V_fit:.2f}\n  τ = {tau_fit:.3f} ± {tau_err:.3f}')
plt.xlabel('tempo (μs)')
plt.ylabel('voltaggio (V)')
plt.title('circuito RL: Fit della Scarica')
plt.grid(True)
plt.legend()

# Annotazioni
textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
plt.text(0.05, max(voltaggi)*0.7, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
#estrapolazione induttanza scarica:
inductance_scarica = tau_fit * Resistance   # in milli Henry
err_inductance_scarica = np.sqrt((tau_err * Resistance)**2 + (tau_fit * err_resistance)**2)  # in milli Henry
print(f"Induttanza Scarica: {inductance_scarica:.2f} ± {err_inductance_scarica:.2f} mH")
#compatibilità tra i due risultati:
n = abs(inductance - inductance_scarica)/ np.sqrt(err_inductance**2 + err_inductance_scarica**2)
print(f"Compatibilità tra i due risultati: {n:.2f} σ")


