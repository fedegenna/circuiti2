
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Nuovo modello: smorzamento armonico classico
def model(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)

def main():
    # Dati
    tempi = np.array([0.08, 0.100, 0.160, 0.240, 0.320, 0.400, 0.480, 0.540, 0.560, 0.640,
                      0.720, 0.750, 0.800, 0.880, 0.970, 1.04, 1.12, 1.2, 1.28, 1.4,
                      1.48, 1.56, 1.65, 1.72, 1.83])
    voltaggi = np.array([101, 109, 72, -30, -78, -32, 42, 62, 60, 12, -42, -47,
                         -40, 6, 38, 24, -14, -26, -8, 23, 12, -10, -15, -2, 12]) - 8
    err_voltaggi = np.sqrt(6**2 + np.array([7,3,4,6,6,4,4,4,4,4,4,3,4,4,4,4,4,4,4,5,4,4,3,4,6])**2)

    # Parametri iniziali realistici
    p0 = [100, 5, 50, 0]

    # Fit
    popt, pcov = curve_fit(model, tempi, voltaggi, sigma=err_voltaggi, absolute_sigma=True, p0=p0)
    A_fit, gamma_fit, omega_fit, phi_fit = popt
    A_err, gamma_err, omega_err, phi_err = np.sqrt(np.diag(pcov))

    # Chi² e p-value
    residuals = voltaggi - model(tempi, *popt)
    chi2_val = np.sum((residuals / err_voltaggi)**2)
    ndof = len(tempi) - len(popt)
    chi2_red = chi2_val / ndof
    p_value = 1 - chi2.cdf(chi2_val, ndof)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(tempi, voltaggi, yerr=err_voltaggi, fmt='o', label='Dati')
    t_fit = np.linspace(min(tempi), max(tempi), 500)
    plt.plot(t_fit, model(t_fit, *popt), 'r-', label=
             f'A = {A_fit:.2f} ± {A_err:.2f}\n'
             f'γ = {gamma_fit:.2f} ± {gamma_err:.2f}\n'
             f'ω = {omega_fit:.2f} ± {omega_err:.2f}\n'
             f'ϕ = {phi_fit:.2f} ± {phi_err:.2f}')

    plt.xlabel('tempo (ms)')
    plt.ylabel('voltaggio (mV)')
    plt.title('Oscillazione smorzata')
    plt.grid(True)
    plt.legend()

    textstr = f"$\chi^2_{{rid}}$ = {chi2_red:.2f}\n$p$-value = {p_value:.3f}"
    plt.text(0.05, max(voltaggi)*0.7, textstr, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()
    C = 97
    L=1/(C*((omega_fit**2)+(gamma_fit**2)))*1000
    print(f"Induttanza: {L:.2f} H")

if __name__ == "__main__":
    main()


