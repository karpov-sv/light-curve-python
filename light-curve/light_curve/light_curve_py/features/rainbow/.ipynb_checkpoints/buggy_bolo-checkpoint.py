'''
def expo(t, t0, rise_time):
    dt = t - t0
    return np.exp(dt / rise_time)
expo_parameter_names = ["reference_time", "rise_time"]

def expo_initial_guesses(baseline, t, m, band):
    initial = {}
    initial["reference_time"] = t[0]
    initial["rise_time"] = 1.0
    return initial
    
def expo_limits(baseline, t, m, band):
    t_amplitude = np.ptp(t)
    limits = {}
    limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
    limits["rise_time"] = (1e-4, 10 * t_amplitude)
    return limits


def linexp(t, t0, amplitude, rise_time):
    dt = t - t0
    return - amplitude * dt * np.exp(dt / rise_time)

linexp_parameter_names = ["reference_time", "amplitude", "rise_time"]

def linexp_initial_guesses(baseline, t, m, band):
    if baseline:
        A = np.ptp(m)
    else:
        A = np.max(m)

    initial = {}
    initial["reference_time"] = t[np.argmax(m)]*0.75
    initial["amplitude"] = A
    initial["rise_time"] = 0.1
    return initial

def linexp_limits(baseline, t, m, band):
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


def doublexp(t, t0, amplitude, p1, p2, p3):
    dt = t - t0
    return amplitude * np.exp(-p1 * dt * (p2 - np.exp(p3 * dt)))

doublexp_parameter_names = ["reference_time", "amplitude", "p1", "p2", "p3"]

def doublexp_initial_guesses(baseline, t, m, band):

        if baseline:
            A = np.ptp(m)
        else:
            A = np.max(m)

        initial = {}
        initial["reference_time"] = t[np.argmax(m)]
        initial["amplitude"] = A
        initial["p1"] = 1.0
        initial["p2"] = 1.0
        initial["p3"] = 1.0
        return initial
    
def doublexp_limits(baseline, t, m, band):

    t_amplitude = np.ptp(t)
    if baseline:
        m_amplitude = np.ptp(m)
    else:
        m_amplitude = np.max(m)

    limits = {}
    limits["reference_time"] = (np.min(t) - 10 * t_amplitude, np.max(t) + 10 * t_amplitude)
    limits["amplitude"] = (0.0, 10 * m_amplitude)
    limits["p1"] = (.01, 10)
    limits["p2"] = (.01, 10)
    limits["p3"] = (.01, 10)
    return limits


bolometric_dict = {'expo' : [expo, expo_parameter_names, expo_initial_guesses, expo_limits],
             'linexp' : [linexp, linexp_parameter_names, linexp_initial_guesses, linexp_limits],
             'doublexp' : [doublexp, doublexp_parameter_names, doublexp_initial_guesses, doublexp_limits]}
'''