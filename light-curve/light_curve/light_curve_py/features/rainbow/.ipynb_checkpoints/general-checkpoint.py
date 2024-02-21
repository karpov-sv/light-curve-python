from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from light_curve.light_curve_py.dataclass_field import dataclass_field
from light_curve.light_curve_py.features.rainbow._base import BaseRainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler, Scaler
import light_curve.light_curve_py.features.rainbow.bolometric as bolo
import light_curve.light_curve_py.features.rainbow.temperature as temp

__all__ = ["RainbowGeneralFit"]


@dataclass()
class RainbowGeneralFit(BaseRainbowFit):
    """Multiband blackbody fit to the light curve using functions to be chosen by the user

    Note, that `m` and corresponded `sigma` are assumed to be flux densities.

    Based on Russeil et al. 2023, arXiv:2310.02916.

    Parameters
    ----------
    band_wave_cm : dict
        Dictionary of band names and their effective wavelengths in cm.

    with_baseline : bool, optional
        Whether to include an offset in the fit, individual for each band.
        If it is true, one more fit paramter per passband is added -
        the additive constant with the same units as input flux.

    with_temperature_evolution : bool, optional
        Whether to include temperature evolution in the fit

    with_rising_only : bool, optional
        If false, only rising part of the Bazin function is used, making
        it effectively a sigmoid function

    Methods
    -------
    __call__(t, m, sigma, band, *, sorted=False, check=True, fill_value=None)
        Evaluate the feature. Positional arguments are numpy arrays of the same length,
        `band` must consist of the same strings as keys in `band_wave_cm`. If `sorted` is True,
        `t` must be sorted in ascending order. If `check` is True, the input is checked for
        NaNs and Infs. If `fill_value` is not None, it is used to fill the output array if
        the feature cannot be evaluated.

    model(t, band, *params)
        Evaluate Rainbow model on the given arrays of times and bands. `*params` are
        fit parameters, basically the output of `__call__` method but without the last
        parameter (reduced Chi^2 of the fit). See parameter names in the `.name` attribute.

    peak_time(*params)
        Return bolometric peak time for given set of parameters
    """

    bolometric: str = dataclass_field(default='bazin', kw_only=True)
    """Which parametric bolometric flux to use"""
    temperature: str = dataclass_field(default='logistic', kw_only=True)
    """Which parametric temperature to use"""

    def _common_parameter_names(self) -> List[str]:
        bolometric_parameters = bolo.bolometric_dict[self.bolometric][1]
        temperature_parameters = temp.temperature_dict[self.temperature][1]
        return [j for j in bolometric_parameters if j in temperature_parameters]
    
    def _bolometric_parameter_names(self) -> List[str]:
        bolometric_parameters = bolo.bolometric_dict[self.bolometric][1]
        return [i for i in bolometric_parameters if i not in self._common_parameter_names()]

    def _temperature_parameter_names(self) -> List[str]:
        temperature_parameters = temp.temperature_dict[self.temperature][1]
        return [i for i in temperature_parameters if i not in self._common_parameter_names()]

    def bol_func(self, t, params):
        return bolo.bolometric_dict[self.bolometric][0](t, *params[self.p.all_bol_idx])

    def temp_func(self, t, params):
        return temp.temperature_dict[self.temperature][0](t, *params[self.p.all_temp_idx])

    def _normalize_bolometric_flux(self, params) -> None:
        if hasattr(self.p, 'amplitude'):
            # Internally we use amplitude of F_bol / <nu> instead of F_bol.
            # It makes amplitude to be in the same units and the same order as
            # the baselines and input fluxes.
            params[self.p.amplitude] /= self.average_nu

    def _denormalize_bolometric_flux(self, params) -> None:
        if hasattr(self.p, 'amplitude'):
            params[self.p.amplitude] *= self.average_nu

    def _unscale_parameters(self, params, t_scaler: Scaler, m_scaler: MultiBandScaler) -> None:
        
        self._denormalize_bolometric_flux(params)
        
        # For sure there should be a denser way to write this
                  
        if hasattr(self.p, 'reference_time'):
            params[self.p.reference_time] = t_scaler.undo_shift_scale(params[self.p.reference_time])

        if hasattr(self.p, 'rise_time'):
            params[self.p.rise_time] = t_scaler.undo_scale(params[self.p.rise_time])
            
        if hasattr(self.p, 'fall_time'):
            params[self.p.fall_time] = t_scaler.undo_scale(params[self.p.fall_time])
            
        if hasattr(self.p, 'k_sig'):
            params[self.p.k_sig] = t_scaler.undo_scale(params[self.p.k_sig])
                
        if hasattr(self.p, 'amplitude'):       
            params[self.p.amplitude] = m_scaler.undo_scale(params[self.p.amplitude])

    def _initial_guesses(self, t, m, band) -> Dict[str, float]:
        initial_bolo = bolo.bolometric_dict[self.bolometric][2](self.with_baseline, t, m, band)
        initial_temp = temp.temperature_dict[self.temperature][2](t, m, band)
        return initial_bolo | initial_temp

    def _limits(self, t, m, band) -> Dict[str, Tuple[float, float]]:
        limits_bolo = bolo.bolometric_dict[self.bolometric][3](self.with_baseline, t, m, band)
        limits_temp = temp.temperature_dict[self.temperature][3](t, m, band)        
        return limits_bolo | limits_temp

    def _baseline_initial_guesses(self, t, m, band) -> Dict[str, float]:
        """Initial guesses for the baseline parameters."""
        return {self.p.baseline_parameter_name(b): np.median(m[band == b]) for b in self.bands.names}

    def peak_time(self, params) -> float:
        """Returns true bolometric peak position for given parameters"""
        # Not correct, should be computed in a different way for each function
        return params[self.p.reference_time]

