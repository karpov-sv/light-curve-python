import numpy as np


def constant(t, T):
    return T
constant_parameter_names = ['T']

def constant_initial_guesses(t, m, band):
    initial = {}
    initial["T"] = 8000.0
    return initial

def constant_limits(t, m, band):
    limits = {}
    limits["T"] = (1e2, 2e6)  # K
    return limits

def logistic(t, t0, Tmin, Tmax, k_sig):
    dt = t - t0
    result = np.zeros_like(dt)

    # To avoid numerical overflows, let's only compute the exponent not too far from t0
    idx1 = dt <= -100 * k_sig
    idx2 = (dt > -100 * k_sig) & (dt < 100 * k_sig)
    idx3 = dt >= 100 * k_sig

    result[idx1] = Tmin
    result[idx2] = Tmin + (Tmax - Tmin) / (1.0 + np.exp(dt[idx2] / k_sig))
    result[idx3] = Tmax
    return result

logistic_parameter_names = ['reference_time', "Tmin", "Tmax", "k_sig"]

def logistic_initial_guesses(t, m, band):
    initial = {}
    initial["Tmin"] = 4000.0
    initial["Tmax"] = 10000.0
    initial["k_sig"] = 1.0
    return initial


def logistic_limits(t, m, band):
    t_amplitude = np.ptp(t)
    limits = {}
    limits["Tmin"] = (1e2, 1e6)  # K
    limits["Tmax"] = (1e2, 1e6)  # K
    limits["k_sig"] = (1e-4, 10 * t_amplitude) # K
    return limits
            
temperature_dict = {'constant' : [constant, constant_parameter_names, constant_initial_guesses, constant_limits],
              'logistic' : [logistic, logistic_parameter_names, logistic_initial_guesses, logistic_limits]}




