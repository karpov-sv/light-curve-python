import numpy as np

def sigmoid(t, t0, amplitude, rise_time):
    dt = t - t0
    result = np.zeros_like(dt)
    # To avoid numerical overflows, let's only compute the exponents not too far from t0
    idx = dt > -100 * rise_time
    result[idx] = amplitude / (np.exp(-dt[idx] / rise_time) + 1)
    return result

sigmoid_parameter_names = ["reference_time", "amplitude", "rise_time"]

def sigmoid_initial_guesses(baseline, t, m, band):
    if baseline:
        A = np.ptp(m)
    else:
        A = np.max(m)

    initial = {}
    initial["reference_time"] = t[np.argmax(m)]
    initial["amplitude"] = A
    initial["rise_time"] = 1.0
    return initial

    
def sigmoid_limits(baseline, t, m, band):
    t_amplitude = np.ptp(t)
    if baseline:
        m_amplitude = np.ptp(m)
    else:
        m_amplitude = np.max(m)

    limits = {}
    limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
    limits["amplitude"] = (0.0, 10 * m_amplitude)
    limits["rise_time"] = (1e-4, 10 * t_amplitude)
    return limits

def bazin(t, t0, amplitude, rise_time, fall_time):
    dt = t - t0
    result = np.zeros_like(dt)
    idx = (dt > -100 * rise_time) & (dt < 100 * fall_time)
    result[idx] = amplitude / (np.exp(-dt[idx] / rise_time) + np.exp(dt[idx] / fall_time))
    return result

bazin_parameter_names = ["reference_time", "amplitude", "rise_time", "fall_time"]

def bazin_initial_guesses(baseline, t, m, band):
    if baseline:
        A = np.ptp(m)
    else:
        A = np.max(m)
        
    initial = {}
    initial["reference_time"] = t[np.argmax(m)]
    initial["amplitude"] = A
    initial["rise_time"] = 0.1
    initial["fall_time"] = 0.1
    return initial
            
def bazin_limits(baseline, t, m, band):
    t_amplitude = np.ptp(t)
    if baseline:
        m_amplitude = np.ptp(m)
    else:
        m_amplitude = np.max(m)

    limits = {}
    limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
    limits["amplitude"] = (0.0, 10 * m_amplitude)
    limits["rise_time"] = (1e-4, 10 * t_amplitude)
    limits["fall_time"] = (1e-4, 10 * t_amplitude)
    return limits


bolometric_dict = {'sigmoid' : [sigmoid, sigmoid_parameter_names, sigmoid_initial_guesses, sigmoid_limits],
                 'bazin' : [bazin, bazin_parameter_names, bazin_initial_guesses, bazin_limits]}