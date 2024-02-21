import sys

import numpy as np
import pytest

from light_curve.light_curve_py import RainbowFit


def bb_nu(wave_aa, T):
    nu = 3e10 / (wave_aa * 1e-8)
    return 2 * 6.626e-27 * nu**3 / 3e10**2 / np.expm1(6.626e-27 * nu / (1.38e-16 * T))


@pytest.mark.skipif(sys.version_info < (3, 8), reason="iminuit requires Python >= 3.8")
def test_noisy_no_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    fall_time = 30.0
    Tmin = 5e3
    delta_T = 10e3
    k_sig = 4.0

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time + 3 * fall_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))
    waves = np.array([band_wave_aa[b] for b in band])

    temp = Tmin + delta_T / (1.0 + np.exp((t - reference_time) / k_sig))
    lum = amplitude * np.exp(-(t - reference_time) / fall_time) / (1.0 + np.exp(-(t - reference_time) / rise_time))

    flux = np.pi * bb_nu(waves, temp) / (5.67e-5 * temp**4) * lum
    # S/N = 5 for minimum flux, scale for Poisson noise
    flux_err = np.sqrt(flux * np.min(flux) / 5.0)
    flux += rng.normal(0.0, flux_err)

    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=False)

    expected = [reference_time, amplitude, rise_time, fall_time, Tmin, delta_T, k_sig, 1.0]
    actual = feature(t, flux, sigma=flux_err, band=band)

    # import matplotlib.pyplot as plt
    # plt.scatter(t, flux, s=5, label="data")
    # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
    # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
    # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(actual, expected, rtol=0.1)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="iminuit requires Python >= 3.8")
def test_noisy_with_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}
    average_nu = np.mean([3e10 / (w * 1e-8) for w in band_wave_aa.values()])

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 10.0
    fall_time = 30.0
    Tmin = 10e3
    delta_T = 5e3
    k_sig = 10.0
    baselines = {b: rng.exponential(scale=3 * amplitude / average_nu) for b in band_wave_aa}

    t = np.sort(rng.uniform(reference_time - 5 * rise_time, reference_time + 5 * fall_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))
    waves = np.array([band_wave_aa[b] for b in band])

    temp = Tmin + delta_T / (1.0 + np.exp((t - reference_time) / k_sig))
    lum = amplitude * np.exp(-(t - reference_time) / fall_time) / (1.0 + np.exp(-(t - reference_time) / rise_time))

    flux = np.pi * bb_nu(waves, temp) / (5.67e-5 * temp**4) * lum + np.array([baselines[b] for b in band])
    # S/N = 5 for minimum flux, scale for Poisson noise
    # We make noise a bit smaller because the optimization is not perfect
    flux_err = 0.1 * np.sqrt(flux * np.min(flux) / 5.0)
    flux += rng.normal(0.0, flux_err)

    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True)

    expected = [reference_time, amplitude, rise_time, fall_time, Tmin, delta_T, k_sig, *baselines.values(), 1.0]
    actual = feature(t, flux, sigma=flux_err, band=band)

    # import matplotlib.pyplot as plt
    # plt.scatter(t, flux, s=1, label="data")
    # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
    # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
    # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(actual, expected, rtol=0.1)
