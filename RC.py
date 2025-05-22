import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2
from iminuit.cost import LeastSquares

def exp(t, V_0, tau):
    return V_0*np.exp(-t / (tau))

def main():

    tempi = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.72, 2.04, 2.36, 2.64, 3.00, 3.40, 3.8, 4.2, 4.6, 5, 5.4, 5.80, 6.2, 6.6, 7, 8, 9])
    voltaggi = np.array([3.9, 3.58, 3.26, 2.98, 2.74, 2.46, 2.22, 2.04, 1.74, 1.50, 1.28, 1.12, 0.96, 0.8, 0.68, 0.56, 0.46, 0.38, 0.32, 0.26, 0.22, 0.2, 0.18, 0.1, 0.06])
    err_voltaggi = 0.02*np.ones([len(voltaggi)])
    my_cost_func = LeastSquares(tempi, voltaggi, err_voltaggi, exp)
    m = Minuit(my_cost_func, V_0=3.98, tau= 2)
    m.migrad()

    # Estrazione risultati
    V_0_fit = m.values['V_0']
    tau_fit = m.values['tau']
    V_0_err = m.errors['V_0']
    tau_err = m.errors['tau']
    chi2_val = m.fval
    ndof = len(tempi) - 1  # un solo parametro
    chi2_red = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)


    plt.figure(figsize=(10, 6))
    plt.errorbar(tempi, voltaggi, yerr=err_voltaggi, fmt='o', label='Dati')
    plt.plot(tempi, exp(tempi, V_0_fit,tau_fit), 'r-', label=f' V_0 = {V_0_fit:.3f} ± {V_0_err:.3f} \n tau = {tau_fit:.3f} ± {tau_err:.3f}')
    plt.xlabel('tempo (ms)')
    plt.ylabel('voltaggio (V)')
    plt.title('Fit: y = V_0·exp(-t/tau)')
    plt.grid(True)
    plt.legend()

    # Annotazioni
    textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
    plt.text(0.05, max(voltaggi)*0.7, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    R = 21.78 * 10**3
    C = tau_fit/R * 10**(6)
    print(f"Capacitance: {C:.2f} nF")
if __name__ == "__main__":
    main()



