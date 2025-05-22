import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Modello sovrasmorzato
def sovrasmorzato(t, A, gamma, omega0):
    # Protezione contro radice di numero negativo
    delta = gamma**2 - omega0**2
    if delta <= 0:
        return np.full_like(t, np.nan)  # restituisce nan se non sovrasmorzato
    beta = np.sqrt(delta)
    return A * (np.exp(-(gamma - beta) * t) - np.exp(-(gamma + beta) * t))

def main():
    # Dati
    tempi = np.array([4, 10, 14, 20, 24, 30, 34, 38, 46, 50, 60, 70, 88, 110, 136,
                      166, 188, 238, 292, 342, 392, 440, 474, 522, 572, 748, 870, 986])
    voltaggi = np.array([0.960, 2.14, 2.62, 3.08, 3.3, 3.48, 3.54, 3.57, 3.59, 3.57,
                         3.52, 3.45, 3.32, 3.13, 2.93, 2.72, 2.57, 2.26, 1.96, 1.72,
                         1.52, 1.35, 1.23, 1.09, 0.96, 0.620, 0.46, 0.34]) - 0.020
    err_misura = np.array([0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
                           0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.02,
                           0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
    err_voltaggi = np.sqrt(0.020**2 + err_misura**2)

    # Stime iniziali
    p0 = [2.0, 0.050, 0.014]  # A, gamma, omega0
    bounds = ([0, 0, 0], [10, 1, 1])  # vincoli positivi realistici

    # Fit
    popt, pcov = curve_fit(
        sovrasmorzato, tempi, voltaggi,
        sigma=err_voltaggi, absolute_sigma=True,
        p0=p0, bounds=bounds, maxfev=10000
    )
    A_fit, gamma_fit, omega0_fit = popt
    A_err, gamma_err, omega0_err = np.sqrt(np.diag(pcov))

    # Chi² e p-value
    residuals = voltaggi - sovrasmorzato(tempi, *popt)
    chi2_val = np.sum((residuals / err_voltaggi)**2)
    ndof = len(tempi) - len(popt)
    chi2_red = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(tempi, voltaggi, yerr=err_voltaggi, fmt='o', label='Dati')
    t_fit = np.linspace(min(tempi), max(tempi), 500)
    plt.plot(t_fit, sovrasmorzato(t_fit, *popt), 'r-', label=
             f'A = {A_fit:.3f} ± {A_err:.3f}\n'
             f'γ = {gamma_fit:.5f} ± {gamma_err:.5f}\n'
             f'ω₀ = {omega0_fit:.5f} ± {omega0_err:.5f}')

    plt.xlabel('tempo (ms)')
    plt.ylabel('voltaggio (V)')
    plt.title('Fit con modello sovrasmorzato')
    plt.grid(True)
    plt.legend()

    textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
    plt.text(tempi[0], max(voltaggi)*0.8, textstr, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
