import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2
from iminuit.cost import LeastSquares

def exp(t, V_0, tau):
    return V_0*(1-np.exp(-t / (tau)))

def main():

    tempi = np.array([6, 14, 22, 30, 38, 46, 56, 62, 70, 78, 86, 94, 102, 118, 130, 156, 180, 206])
    voltaggi = 1.80 + np.array([-1.40, -0.880, -0.440, -0.08, 0.24, 0.48, 0.72, 0.92, 1.08, 1.20, 1.28, 1.36, 1.44, 1.56, 1.64, 1.72, 1.80, 1.84])
    err_voltaggi = np.array([np.sqrt(2)*0.04, np.sqrt(5)*0.08, np.sqrt(2)*0.04, np.sqrt(5)*0.08, np.sqrt(5)*0.08, np.sqrt(5)*0.08, np.sqrt(5)*0.08, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(5)*0.08, np.sqrt(5)*0.08, np.sqrt(5)*0.08, np.sqrt(5)*0.08, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(2)*0.04, np.sqrt(5)*0.08])
    my_cost_func = LeastSquares(tempi, voltaggi, err_voltaggi, exp)
    m = Minuit(my_cost_func, V_0=3.64, tau= 22)
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
    plt.plot(tempi, exp(tempi, V_0_fit,tau_fit), 'r-', label=f' $V_0$ = {V_0_fit:.3f} ± {V_0_err:.3f} \n τ = {tau_fit:.3f} ± {tau_err:.3f}')
    plt.xlabel('tempo (μs)')
    plt.ylabel('voltaggio (V)')
    #plt.title('Fit: y = V_0·(1-exp(-t/tau))')
    plt.grid(True)
    plt.legend()

    # Annotazioni
    textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
    plt.text(0.05, max(voltaggi)*0.7, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    R = 0.995 * 10**3
    L = tau_fit*R * 10**(-6)
    print(f"Induttanza: {L:.2f} ")

if __name__ == "__main__":
    main()
