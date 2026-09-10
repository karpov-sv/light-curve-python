from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

__all__ = [
    "spectral_terms",
    "BaseSpectralTerm",
    "PlanckSpectralTerm",
    "BlanketedPlanckSpectralTerm",
    "FreeBlanketedPlanckSpectralTerm",
    "GenWienSpectralTerm",
    "ModifiedBlackBodySpectralTerm",
    "SharpBlackBodySpectralTerm",
    "LogParabolaSpectralTerm",
]

# CODATA 2018
planck_constant = 6.62607004e-27  # erg s
speed_of_light = 2.99792458e10  # cm/s
boltzman_constant = 1.380649e-16  # erg/K
b_wien = 28977720  # Angstrom*K


@dataclass()
class BaseSpectralTerm:
    """Spectral term for Rainbow"""

    @staticmethod
    @abstractmethod
    def parameter_names() -> List[str]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def parameter_scalings() -> List[Union[str, None]]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def value(wave_cm, T, *params):
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def initial_guesses(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def limits(t, m, sigma, band) -> Dict[str, float]:
        return NotImplementedError

    @staticmethod
    def parameter_priors() -> Dict[str, Tuple[float, float]]:
        """Optional Gaussian priors ``{name: (mean, sigma)}`` on parameters.

        Only parameters with ``None`` scaling are supported (the prior is applied
        directly in the fit's scaled space, which for unscaled parameters equals the
        physical value). Default: no priors.
        """
        return {}


@dataclass()
class PlanckSpectralTerm(BaseSpectralTerm):
    """Standard blackbody spectrum"""

    @staticmethod
    def parameter_names():
        return []

    @staticmethod
    def parameter_scalings():
        return []

    @staticmethod
    def value(wave_cm, T, *params):
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        # B = (2h/c²) ν³ / (e^x - 1), written as ν³ e^{-x} / (1 - e^{-x}) to stay overflow-safe
        # at large x (cold T / blue wavelengths): e^x / expm1(x) overflows there, whereas e^{-x}
        # underflows to 0 and B → 0 (the Wien tail), which is the correct limit.
        return (2 * planck_constant / speed_of_light**2) * nu**3 * np.exp(-x) / (-np.expm1(-x))

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {}

    @staticmethod
    def limits(t, m, sigma, band):
        return {}

    @staticmethod
    def dvalue_dT(wave_cm, T, *params):
        """∂(planck)/∂T. `*params` is empty for plain Planck (kept for API parity)."""
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        neg_em1 = -np.expm1(-x)  # 1 - e^{-x}, the stable denominator (no e^{+x})
        planck = (2 * planck_constant / speed_of_light**2) * nu**3 * np.exp(-x) / neg_em1
        # ∂planck/∂T = planck · x·e^x/(T·expm1(x)) = planck · x / (T · (1 - e^{-x}))
        return planck * x / (T * neg_em1)

    @staticmethod
    def derivatives(wave_cm, T, *params):
        """No spectral parameters; returns shape (0, len(wave_cm))."""
        return np.zeros((0, len(wave_cm)))


@dataclass()
class BlanketedPlanckSpectralTerm(BaseSpectralTerm):
    r"""Blackbody spectrum with exponential UV blanketing.

    .. math::
        F(\lambda) = B_\nu(\lambda, T)\, e^{-\tau},\quad
        \tau = I\, e^{-\lambda / \lambda_s},\quad
        \lambda_s = \texttt{max\_extinction}\cdot \texttt{lambda\_scale} / T_\mathrm{ref}

    The blackbody core follows the instantaneous temperature ``T(t)``, but the *depth* of
    the UV extinction should not vary as the source cools, so the extinction reach
    ``lambda_s`` is anchored to a single, time-constant characteristic temperature
    ``T_ref`` rather than to the instantaneous one.

    That anchor is the temperature term's characteristic ``T``, which this term **shares**
    via the ``common_temp_spec`` machinery (so the only new fit parameter is
    ``lambda_scale``):

    - with ``temperature='constant'`` the characteristic ``T`` is the (constant)
      temperature itself, so ``T_ref`` equals the instantaneous ``T``;
    - with ``temperature='sigmoid'`` / ``'delayed_sigmoid'`` it is the mid temperature
      ``(Tmin + Tmax) / 2``, so the blanketing depth is pinned to that characteristic value
      while the Planck core still cools.

    A single implementation therefore covers every temperature term; the value of the
    shared ``T`` simply differs (constant vs. mid).
    """

    # Fixed blanketing intensity and extinction reach; referenced by `value` and the gradients.
    _intensity = 100
    _max_extinction = 2 * b_wien

    @staticmethod
    def parameter_names():
        # ``T`` is shared with the temperature term; only ``lambda_scale`` is a new parameter.
        return ["T", "lambda_scale"]

    @staticmethod
    def parameter_scalings():
        return [None, None]

    @staticmethod
    def value(wave_cm, T, T_ref, lambda_scale):
        # ``T`` is the instantaneous temperature (cooling, for sigmoid terms); ``T_ref`` is
        # the shared, time-constant characteristic temperature anchoring the extinction reach
        # (equal to ``T`` for a constant temperature term).
        base = PlanckSpectralTerm.value(wave_cm, T)

        intensity = BlanketedPlanckSpectralTerm._intensity

        # Encodes how far (in wavelength) the maximum UV extinction extends (at lambda_scale=1).
        # Allows the extinction to affect the BB wavelength past the peak (in the formula below).
        max_extinction = BlanketedPlanckSpectralTerm._max_extinction

        # Lambda_angstrom represents how far (in absolute wavelength) the extinction affects the BB.
        # Lambda_scale quantifies this between 0 (no UV ext) and 1 max suppression (encoded by max_extinction above).
        # Anchored to the constant T_ref so the blanketing depth does not vary as the object cools.
        lambda_angstrom = max_extinction * lambda_scale / T_ref

        # Convert to cm to match wave_cm
        lambda_cm = lambda_angstrom * 1e-8

        tau = intensity * np.exp(-wave_cm / lambda_cm)

        return base * np.exp(-tau)

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        # ``T`` initial guess/limits come from the (shared) temperature term.
        # Start strictly inside the bounds with a *live* gradient: at the lower bound
        # (lambda_scale = 0.001) the extinction edge sits at a few Angstrom, tau is
        # numerically 0 in every optical band and the gradient vanishes — a flat plateau
        # the optimizer must escape before fitting anything. 0.25 is the median regime of
        # typical fitted values (edge in the near-UV/blue); starts below ~0.1 leave the
        # gradient too weak to reliably steer some term combinations to the right basin.
        return {
            "lambda_scale": 0.25,
        }

    @staticmethod
    def limits(t, m, sigma, band):
        return {
            "lambda_scale": (0.001, 1.0),
        }

    @staticmethod
    def _planck_and_tau(wave_cm, T, T_ref, lambda_scale):
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        # Overflow-safe Planck core (see PlanckSpectralTerm.value): use e^{-x} / (1 - e^{-x})
        # instead of 1 / expm1(x) so cold T / blue wavelengths give 0, not inf/nan.
        neg_em1 = -np.expm1(-x)  # 1 - e^{-x}
        planck = (2 * planck_constant / speed_of_light**2) * nu**3 * np.exp(-x) / neg_em1
        # Extinction reach anchored to the constant characteristic temperature T_ref (see `value`).
        lambda_cm = BlanketedPlanckSpectralTerm._max_extinction * lambda_scale / T_ref * 1e-8
        u = wave_cm / lambda_cm
        tau = BlanketedPlanckSpectralTerm._intensity * np.exp(-u)
        return planck, neg_em1, x, tau, u

    @staticmethod
    def dvalue_dT(wave_cm, T, T_ref, lambda_scale):
        """∂(spec)/∂T w.r.t. the *instantaneous* temperature only.

        The extinction depends on the constant anchor ``T_ref``, not on the instantaneous
        ``T``, so only the Planck core varies here; the ``T_ref`` contribution is reported by
        `derivatives` (it is the shared spectral parameter). When the temperature term is
        constant, ``T`` and ``T_ref`` are the same parameter and the two contributions are
        summed by the Jacobian assembly (``jac[idx] += ...``).
        """
        planck, neg_em1, x, tau, _u = BlanketedPlanckSpectralTerm._planck_and_tau(wave_cm, T, T_ref, lambda_scale)
        # ∂planck/∂T = planck · x / (T · (1 - e^{-x})) — the overflow-safe form of x·e^x/(T·expm1(x)).
        dplanck_dT = planck * x / (T * neg_em1)
        return dplanck_dT * np.exp(-tau)

    @staticmethod
    def derivatives(wave_cm, T, T_ref, lambda_scale):
        """∂(spec)/∂(T_ref, lambda_scale); shape (2, len(wave_cm)).

        Row order matches ``parameter_names()`` = ``[T, lambda_scale]`` (the shared ``T``
        enters here as the anchor ``T_ref``). With ``u = wave_cm/λ_cm ∝ T_ref/λ_scale``:
        ``∂u/∂T_ref = u/T_ref`` and ``∂u/∂λ_scale = -u/λ_scale``, and ``∂τ = -τ·∂u``.
        """
        planck, _em1, _x, tau, u = BlanketedPlanckSpectralTerm._planck_and_tau(wave_cm, T, T_ref, lambda_scale)
        spec = planck * np.exp(-tau)
        jac = np.zeros((2, len(wave_cm)))
        # ∂spec/∂T_ref = -spec·∂τ/∂T_ref = +spec·τ·u/T_ref
        jac[0] = spec * tau * u / T_ref
        # ∂spec/∂λ_scale = -spec·∂τ/∂λ_scale = -spec·τ·u/λ_scale
        jac[1] = -spec * tau * u / lambda_scale
        return jac


@dataclass()
class FreeBlanketedPlanckSpectralTerm(BaseSpectralTerm):
    r"""Blackbody with a free-position blue blanketing edge (redshift-free).

    .. math::
        F(\lambda) = B_\nu(\lambda, T)\, e^{-\tau},\qquad
        \tau = I\, e^{-\lambda / \lambda_s}\ \text{(fixed depth)},\qquad
        \tau = \tau_\mathrm{ref}\, e^{(\lambda_\mathrm{ref} - \lambda)/\lambda_s}\ \text{(free depth)}

    Same sharp blue-cutoff family as :class:`BlanketedPlanckSpectralTerm` (fully suppressed
    blueward, Planck redward, with the cutoff near ``lambda_edge ~ ln(I)·lambda_s``), but the
    edge scale ``lambda_s`` — parameter ``blanket_scale``, in angstrom — is a **free
    observed-frame** quantity rather than ``∝ lambda_scale / T``.

    Decoupling the edge from temperature is the whole point: the ``∝ lambda_scale/T`` coupling
    of the original blanketed model makes the edge position and ``T`` enter only through their
    ratio, which is what creates the strong ``(T, lambda_scale)`` degeneracy (median ``|ρ|``
    ≈ 0.97 on mallorn supernovae). Letting the edge roam freely instead drops that to ``|ρ|``
    ≈ 0.44 at no fit-quality cost, and needs **no redshift** — a free observed-frame edge
    absorbs both the rest-frame break and the unknown ``z`` into one fitted number. The red
    continuum is untouched, so ``T`` stays pinned by the red bands.

    Parameters
    ----------
    free_depth : bool, optional
        If False (default) the optical depth ``I`` is fixed at ``_intensity`` (a hard blue
        blackout; the only fit parameter is ``blanket_scale``, carrying a prior
        ``N(scale_prior_mean_aa, scale_prior_sigma_aa)`` anchored at the *sensitivity
        edge*, which parks unblanketed sources at the limit below which the model is Planck
        in every band, rather than in the degenerate blackbody-mimicking twin at ~650 A).
        If True, the depth becomes a
        second fit parameter ``blanket_depth`` (``≥ 0``, one-sided so it can only *suppress*
        the blue, never enhance it), with a weak prior anchoring it to 0 (no blanketing).
        Freeing the depth lets shallow-deficit sources use a soft edge instead of a forced
        blackout, avoiding the temperature overheating that a fixed full depth induces.

        ``blanket_depth`` is **anchored at a fixed reference wavelength**: it is the optical
        depth ``tau`` at ``lambda_ref = lam_ref_aa`` (~u band), not the ``lambda -> 0``
        extrapolation ``I``. Anchoring at a data-relevant wavelength makes the depth a
        directly observable quantity (the blue-band suppression is ``1 - exp(-blanket_depth)``
        for an edge near the bands) and removes the long ``(scale, depth)`` ridge of the
        ``I``-parametrization, where a small change of ``blanket_scale`` rescaled the
        extrapolated ``I`` by orders of magnitude (population corr ~0.7 between the fitted
        pair on thermal mallorn sources, ~0-0.3 once anchored). Both parameters carry weak
        priors: depth toward 0 (no blanketing; its width ``_depth_prior_sigma`` directly
        trades SN fit quality against temperature fidelity), and ``blanket_scale`` toward a
        *low* anchor (``_scale_prior_mean``): with a free depth the edge position is
        unconstrained whenever ``blanket_depth -> 0``, so unblanketed sources collapse to
        the anchor — well below the ~500-1500 A range of genuinely blanketed sources, which
        keeps them out of the measured distribution (and makes ``blanket_scale`` itself a
        thermal/blanketed discriminant) — while the wide prior is overridden wherever the
        data actually constrain the edge.
    lam_ref_aa : float, optional
        Reference wavelength in angstrom, defaulting to the LSST *u* effective wavelength. It
        anchors ``blanket_depth`` in free-depth mode and must sit at or blueward of the bluest
        band in use (see the class body for why). Lower it for a filter set with bluer coverage.
    scale_prior_mean_aa, scale_prior_sigma_aa : float, optional
        Gaussian prior on ``blanket_scale`` in fixed-depth mode, in angstrom. The defaults are
        tuned for LSST *ugrizy*; the mean marks the sensitivity edge, roughly
        ``lam_blue / 9.2``, below which the model is a plain Planck function in every band. The
        width is set by how much chi2 the degenerate solution buys and is *not* a wavelength, so
        rescaling it with the filter set is usually wrong. Ignored when ``free_depth`` is True.

    Notes
    -----
    Unlike :class:`BlanketedPlanckSpectralTerm`, this term does **not** share ``T`` with the
    temperature term (no ``/T`` coupling), so the blackbody core uses the instantaneous
    temperature only and ``dvalue_dT`` carries no extinction cross-term.
    """

    free_depth: bool = False
    # Reference wavelength (angstrom) anchoring ``blanket_depth`` in free_depth mode: the depth
    # is the optical depth at this wavelength, i.e. a directly observable blue suppression. It
    # MUST sit at (or blueward of) the bluest band: if any band were blueward of the anchor, an
    # ultra-sharp edge squeezed between them would amplify that band's optical depth above
    # ``blanket_depth`` (tau scales as e^{(lam_ref - lam)/lam_s}), letting the fit buy blue
    # suppression while dodging the depth prior. Default is the LSST u effective wavelength;
    # lower it for filter sets with bluer coverage.
    lam_ref_aa: float = 3671.0
    # Fixed-depth scale prior (angstrom), see the block below for how these two are set. They
    # are wavelength-dependent, so a filter set that is not LSST-like may want its own; the
    # mean should stay at or below the bluest band's sensitivity edge, ~``lam_blue / 9.2``.
    scale_prior_mean_aa: float = 500.0
    scale_prior_sigma_aa: float = 200.0

    _intensity = 100.0  # fixed optical depth used when ``free_depth`` is False (hard blackout)
    # Weak N(0, sigma) prior anchoring blanket_depth (= tau at _lam_ref_cm) to 0 (no
    # blanketing). This width is the chi2-vs-temperature-fidelity dial on heavily blanketed
    # sources (mallorn SN Ia, median): 3.0 -> rchi2 0.95 / T 34% hot; 1.5 -> 0.99 / 27%;
    # 0.75 -> 1.06 / 18%. Thermal sources stay clean (depth ~0, T within 1%) throughout.
    _depth_prior_sigma = 0.75
    # Weak prior anchoring the edge position (angstrom) when free_depth is on. Whenever
    # blanket_depth -> 0 (an unblanketed source) the likelihood is exactly flat in
    # blanket_scale, so the prior alone decides where it parks; without it the parameter
    # rails to a bound. The mean is deliberately a *low interior* value, well below the
    # range genuinely blanketed sources occupy (~500-1500 A): unconstrained objects then
    # collapse to ~100 instead of sitting amid real measurements, making blanket_scale
    # itself usable to separate unblanketed from blanketed sources. It must stay interior
    # (not at the lower bound) to avoid on-bound convergence pathologies; the wide sigma
    # makes the pull negligible wherever the data actually constrain the edge.
    _scale_prior_mean = 100.0
    _scale_prior_sigma = 1500.0
    # Fixed-depth mode anchors the scale at the *sensitivity edge* instead, with a much
    # tighter sigma. Without a depth off-switch, thermal (unblanketed) sources have a
    # degenerate "twin" solution at scale ~650-1000 A (hotter blackbody x blue edge ~ cooler
    # blackbody), and the fit has to choose between two wells on tiny chi2 differences.
    #
    # Two facts set these numbers. (1) With the depth fixed at ``_intensity`` the edge sits
    # near ``ln(I)*lambda_s``, so below ~400-500 A the suppression in the bluest band drops
    # under a percent: the model is Planck in every band, the likelihood is exactly flat and
    # every value there describes the same observable spectrum. Anchoring at 100 A spent the
    # prior's pull inside that dead zone, where it buys nothing, and needed a wide sigma to
    # still reach the twin -- which then also dragged down genuine detections. Anchoring at
    # the edge puts the whole gradient where the parameter is actually observable, and makes
    # the parked value an honest upper limit ("edge not detectable") rather than a specific
    # value the data never measured. (2) On mallorn the twin's typical chi2 advantage is
    # ~1, so the prior only has to charge a bit more than that at ~700 A: sigma 200 gives
    # ((700-500)/200)^2 = 1.0 at the twin and only ((900-500)/200)^2 = 4 at a real SN edge,
    # whose median chi2 gain is ~8.
    #
    # Measured on 670 mallorn objects (350 AGN+TDE, 320 SNe), thermal leak above 600 A vs
    # SNe kept above 600 A vs AUC of the SN/thermal separation:
    #   anchor 100, sigma 600 (previous):  13.1% / 81.2% / 0.874, null parks at 113 A
    #   anchor 500, sigma 200 (this):      10.0% / 78.8% / 0.872, null parks at 488 A
    #   anchor 500, sigma 150:              5.1% / 73.1% / 0.869
    #   anchor 100, sigma 350:              4.6% / 73.1% / 0.855  (same leak, worse AUC)
    # Tighten sigma toward 150 when clean thermal nulls matter more than SN completeness;
    # the leak/completeness trade is an information limit (the faintest quartile of SNe gains
    # as little chi2 as the strongest tenth of thermal sources), so no anchor escapes it.
    # NB ``scale_prior_mean_aa`` / ``scale_prior_sigma_aa`` apply to fixed-depth mode only; the
    # free-depth branch keeps the low anchor above, where the depth parameter -- not the scale --
    # carries the null.

    @property
    def _lam_ref_cm(self):
        return self.lam_ref_aa * 1e-8

    def parameter_names(self):
        return ["blanket_scale", "blanket_depth"] if self.free_depth else ["blanket_scale"]

    def parameter_scalings(self):
        return [None, None] if self.free_depth else [None]

    def _tau(self, wave_cm, params):
        """Optical depth and its derivative w.r.t. ``blanket_depth`` (None in fixed mode).

        Fixed mode: ``tau = I·e^{-lambda/lambda_s}``. Free mode (anchored): ``tau =
        depth·e^{(lambda_ref - lambda)/lambda_s}``; the exponent is clipped so that
        wavelengths far blueward of the reference with a tiny scale cannot overflow — the
        suppression there is total anyway (``value`` underflows to 0) and the clipped
        derivatives stay consistently 0.
        """
        lam_cm = params[0] * 1e-8  # blanket_scale is in angstrom
        if self.free_depth:
            dtau_ddepth = np.exp(np.minimum((self._lam_ref_cm - wave_cm) / lam_cm, 50.0))
            return params[1] * dtau_ddepth, dtau_ddepth
        return self._intensity * np.exp(-wave_cm / lam_cm), None

    def value(self, wave_cm, T, *params):
        tau, _ = self._tau(wave_cm, params)
        return PlanckSpectralTerm.value(wave_cm, T) * np.exp(-tau)

    def initial_guesses(self, t, m, sigma, band):
        if self.free_depth:
            # Start mid-range, NOT at the low prior anchor: from a small scale the suppression
            # of every band redward of the reference is exponentially tiny, so the gradient
            # toward a broad (g-band-reaching) edge vanishes and blanketed sources would get
            # stranded in a u-only local minimum. From mid-range both directions have signal;
            # unblanketed sources still slide down the prior gradient to the anchor.
            return {"blanket_scale": 700.0, "blanket_depth": 0.1}
        # Mid-range for the same stranding reason: at the prior anchor the in-band suppression
        # (and so the likelihood gradient) is ~e^{-37}, invisible next to even the weak prior
        # gradient, and every fit would get trapped at the anchor.
        return {"blanket_scale": 700.0}

    def limits(self, t, m, sigma, band):
        limits = {"blanket_scale": (20.0, 3000.0)}  # edge ~ ln(I)*lambda_s ~ 90..13800 A
        if self.free_depth:
            # tau_ref > ~20 is already a saturated blackout at the reference wavelength
            # (exp(-20) ~ 0), so a wider bound buys nothing physically while widening the
            # iminuit lower-bound barrier, which repels depth from its null (0) and floors
            # unblanketed sources at ~(1e-4 * width * sigma_prior^2)^(1/3). The least_squares
            # backend has no barrier and reaches depth -> 0 regardless.
            limits["blanket_depth"] = (0.0, 20.0)
        return limits

    def parameter_priors(self):
        cls = FreeBlanketedPlanckSpectralTerm
        if self.free_depth:
            # depth -> 0 anchors the "is it blanketed" amount; the scale prior only stabilizes
            # the position when depth is small (it is a no-op once the data constrain the edge).
            return {
                "blanket_depth": (0.0, cls._depth_prior_sigma),
                "blanket_scale": (cls._scale_prior_mean, cls._scale_prior_sigma),
            }
        # Fixed depth: scale prior anchored at the sensitivity edge, tight enough to tip
        # unblanketed sources out of the degenerate twin basin (see scale_prior_mean_aa).
        return {"blanket_scale": (self.scale_prior_mean_aa, self.scale_prior_sigma_aa)}

    def dvalue_dT(self, wave_cm, T, *params):
        """∂(spec)/∂T w.r.t. the instantaneous temperature only (the edge is T-independent)."""
        tau, _ = self._tau(wave_cm, params)
        return PlanckSpectralTerm.dvalue_dT(wave_cm, T) * np.exp(-tau)

    def derivatives(self, wave_cm, T, *params):
        """∂(spec)/∂(blanket_scale[, blanket_depth]); shape (n_params, len(wave_cm))."""
        lam_cm = params[0] * 1e-8
        tau, dtau_ddepth = self._tau(wave_cm, params)
        value = PlanckSpectralTerm.value(wave_cm, T) * np.exp(-tau)
        jac = np.zeros((2 if self.free_depth else 1, len(wave_cm)))
        if self.free_depth:
            # τ = depth·e^{(λ_ref-λ)/λ_s}: ∂τ/∂λ_s = τ·(λ-λ_ref)/λ_s²; blanket_scale in Å (×1e-8)
            jac[0] = -value * tau * (wave_cm - self._lam_ref_cm) / lam_cm**2 * 1e-8
            # ∂τ/∂depth = e^{(λ_ref-λ)/λ_s}
            jac[1] = -value * dtau_ddepth
        else:
            # τ = I·e^{-λ/λ_s}: ∂τ/∂λ_s = τ·λ/λ_s²
            jac[0] = -value * tau * wave_cm / lam_cm**2 * 1e-8
        return jac


@dataclass()
class GenWienSpectralTerm(BaseSpectralTerm):
    r"""Generalized-Wien optical SED.

    .. math::
        B(\nu) \propto \nu^3 \, e^{-x^{\texttt{spec\_k}}},\quad x = \frac{h\nu}{k_B T}

    A single parameter ``spec_k`` (the blue-falloff exponent) unifies blackbody-like
    and UV-blanketed optical SEDs in one smooth form, replacing the separate
    Planck × extinction product:

    - ``spec_k ≈ 1`` is the Wien tail, which equals Planck for cool sources (large ``x``);
    - ``spec_k > 1`` sharpens the blue cutoff, reproducing the UV deficit of hot sources.

    Unlike the blanketed model, the time-variation of the blue suppression is *inherent*
    in the thermal form (it follows ``T(t)`` through ``x``), so ``spec_k`` is a fixed
    per-object shape index rather than a quantity coupled to the instantaneous temperature.

    Cool sources do not constrain ``spec_k`` (it becomes degenerate with ``T``), so a
    Gaussian prior anchors it to the Wien/Planck-like value 1; hot sources, where the data
    do constrain it, override the prior.

    .. warning::
        The fitted ``T`` is **not** a physical (thermodynamic) temperature. Because
        ``spec_k`` and ``T`` trade off, a pure blackbody is recovered at a strongly biased,
        non-monotonic ``T`` (e.g. a 20 kK blackbody fits at ``T ≈ 3 kK``). The bolometric
        normalization (``∝ T⁴``) is correspondingly meaningless, so the recovered amplitude
        is not a physical luminosity. Treat ``(T, spec_k)`` jointly as SED-shape features.

        An approximate blackbody temperature can be recovered from the fitted pair by
        matching the GenWien spectral peak ``ν_peak = (k_B T / h)·(3/spec_k)^(1/spec_k)``
        to the Wien peak of a blackbody:

        .. math::
            T_\mathrm{BB} \approx 0.29 \; T \; (3/\texttt{spec\_k})^{1/\texttt{spec\_k}}

        This is good to ~10-15% only in the warm regime (~5-22 kK). Above ~25 kK both
        ``T`` and ``spec_k`` saturate, so the blackbody temperature is unrecoverable from
        a GenWien fit — use the Planck spectral term if physical temperatures are needed.
    """

    _prior_mean = 1.0
    _prior_sigma = 0.5

    @staticmethod
    def parameter_names():
        return ["spec_k"]

    @staticmethod
    def parameter_scalings():
        return [None]

    @staticmethod
    def _x_value(wave_cm, T, spec_k):
        nu = speed_of_light / wave_cm
        x = planck_constant * nu / (boltzman_constant * T)
        value = (2 * planck_constant / speed_of_light**2) * nu**3 * np.exp(-np.power(x, spec_k))
        return x, value

    @staticmethod
    def value(wave_cm, T, spec_k):
        return GenWienSpectralTerm._x_value(wave_cm, T, spec_k)[1]

    @staticmethod
    def dvalue_dT(wave_cm, T, spec_k):
        """∂(value)/∂T. With ``x = a/T``: ∂x/∂T = -x/T, so ∂value/∂T = value·k·x^k / T."""
        x, value = GenWienSpectralTerm._x_value(wave_cm, T, spec_k)
        return value * spec_k * np.power(x, spec_k) / T

    @staticmethod
    def derivatives(wave_cm, T, spec_k):
        """∂(value)/∂spec_k; shape (1, len(wave_cm)).  ∂value/∂k = -value·x^k·ln(x)."""
        x, value = GenWienSpectralTerm._x_value(wave_cm, T, spec_k)
        jac = np.zeros((1, len(wave_cm)))
        jac[0] = -value * np.power(x, spec_k) * np.log(x)
        return jac

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {"spec_k": GenWienSpectralTerm._prior_mean}

    @staticmethod
    def limits(t, m, sigma, band):
        return {"spec_k": (0.3, 3.0)}

    @staticmethod
    def parameter_priors():
        return {"spec_k": (GenWienSpectralTerm._prior_mean, GenWienSpectralTerm._prior_sigma)}


@dataclass()
class ModifiedBlackBodySpectralTerm(BaseSpectralTerm):
    r"""Modified blackbody: a Planck spectrum tilted by a power law in wavelength.

    .. math::
        F(\lambda) = B_\nu(\lambda, T) \cdot (\lambda / \lambda_\mathrm{ref})^{\beta}

    The single parameter ``beta`` is anchored at the pure-Planck value 0:

    - ``beta = 0`` is **exactly Planck**, so the temperature stays physical — a pure
      blackbody is recovered with ``beta ≈ 0`` and ``T`` to ~1-6% across 5-35 kK;
    - ``beta > 0`` suppresses the blue (a gentle UV deficit / blanketing);
    - ``beta < 0`` enhances the blue, and ``beta`` together with a high ``T`` (whose
      optical Planck is Rayleigh-Jeans, ``∝ ν²``) reproduces a power-law SED
      ``F_ν ∝ ν^{2-\beta}``.

    Because the deviation is a power-law *tilt* of a preserved Planck core (not a reshape
    of the whole SED), ``beta`` and ``T`` are nearly orthogonal — the best-conditioned of
    the deviation terms. The single power-law tilt is, however, too gentle to reproduce
    the very sharpest blue cutoffs (use ``logparabola`` for those).

    For a *true* blackbody the fit sits at ``beta ≈ 0`` and recovers a physical ``T``, but
    for a genuinely non-blackbody SED (e.g. supernovae, whose optical is closer to a power
    law) ``beta`` and ``T`` become degenerate: the tilt can mimic a temperature change, so
    the fit slides ``beta`` toward its bound and ``T`` to an unphysically cold corner. A
    weak Gaussian prior ``N(0, _prior_sigma)`` anchors ``beta`` toward 0, breaking that
    degeneracy — it is overridden where the data demand a real tilt, but stops the runaway
    railing. ``_prior_sigma`` tunes the trade-off (smaller => more physical T, weaker
    deviation capture).
    """

    _wave_ref_cm = 6000e-8  # reference wavelength (~middle of the optical), in cm
    _prior_sigma = 1.0

    @staticmethod
    def parameter_names():
        return ["beta"]

    @staticmethod
    def parameter_scalings():
        return [None]

    @staticmethod
    def value(wave_cm, T, beta):
        tilt = np.power(wave_cm / ModifiedBlackBodySpectralTerm._wave_ref_cm, beta)
        return PlanckSpectralTerm.value(wave_cm, T) * tilt

    @staticmethod
    def dvalue_dT(wave_cm, T, beta):
        """∂(value)/∂T = ∂Planck/∂T · tilt (the tilt is T-independent)."""
        tilt = np.power(wave_cm / ModifiedBlackBodySpectralTerm._wave_ref_cm, beta)
        return PlanckSpectralTerm.dvalue_dT(wave_cm, T) * tilt

    @staticmethod
    def derivatives(wave_cm, T, beta):
        """∂(value)/∂beta = value · ln(λ/λ_ref); shape (1, len(wave_cm))."""
        rel = wave_cm / ModifiedBlackBodySpectralTerm._wave_ref_cm
        value = PlanckSpectralTerm.value(wave_cm, T) * np.power(rel, beta)
        jac = np.zeros((1, len(wave_cm)))
        jac[0] = value * np.log(rel)
        return jac

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {"beta": 0.0}

    @staticmethod
    def limits(t, m, sigma, band):
        return {"beta": (-6.0, 10.0)}

    @staticmethod
    def parameter_priors():
        return {"beta": (0.0, ModifiedBlackBodySpectralTerm._prior_sigma)}


@dataclass()
class SharpBlackBodySpectralTerm(BaseSpectralTerm):
    r"""Planck spectrum with a sharp power-law-opacity blue suppression.

    .. math::
        F(\lambda) = B_\nu(\lambda, T)\, e^{-\tau},\qquad
        \tau = \beta\,\Big(\frac{\lambda_\mathrm{ref}}{\lambda}\Big)^{\texttt{sharpness}}

    A *sharpened* :class:`ModifiedBlackBodySpectralTerm`: the two are one continuous
    family. Expanding the opacity for ``sharpness -> 0`` gives ``tau ~ beta +
    beta·sharpness·ln(lambda_ref/lambda)``; the constant is a grey rescaling absorbed by
    the bolometric amplitude and the rest is exactly the modified-blackbody tilt with a
    rescaled exponent. At the default ``sharpness = 6`` the opacity is instead strongly
    localized to the blue (``tau`` in the z band is ~0.6% of its u-band value), which is
    the point: ``modified_bb``'s logarithmic tilt is the *gentlest* possible opacity,
    spread over all bands, so fitting a sharp blue deficit forces it to distort the red
    side and compensate with an unphysically low temperature (mallorn SNe:
    ``T``/``T_planck`` ~ 0.6, ``beta`` railed negative). Here the red bands stay clean,
    pinning ``T``, and the suppression is carried by ``beta`` alone (mallorn SN Ia:
    ``T``/``T_planck`` 0.62 -> 1.15, chi2 0.98 -> 1.08 — the modest chi2 cost is the price
    of refusing the unphysical cold corner).

    The parameter is named ``beta`` for drop-in consistency with ``modified_bb`` (same
    role: the single blue-deviation strength, 0 = exact Planck), but it is **not**
    numerically comparable — here it is an optical *depth*, not a tilt exponent, and on
    blanketed SNe it lands positive (~+1) where the modified-blackbody tilt rails negative
    (~-2). ``lambda_ref`` sits at the bluest band, so ``beta`` *is* the u-band optical
    depth (blue suppression ``1 - exp(-beta)``) — a directly observable quantity, anchored
    to 0 (exact Planck) by a weak Gaussian prior like the other deviation terms. Thermal
    sources recover ``beta ~ 0`` with ``T`` within 1% of the pure-Planck fit.

    ``sharpness`` is a fixed design constant, not a fit parameter (4-6 perform nearly
    identically on mallorn; sub-1 values reproduce the modified-blackbody behavior,
    including its temperature degeneracy).

    Parameters
    ----------
    sharpness : float, optional
        Exponent of the power-law opacity; how tightly the suppression is confined to the blue.
    lam_ref_aa : float, optional
        Reference wavelength in angstrom, defaulting to the LSST *u* effective wavelength. Set it
        to the bluest band of the filter set in use, so that ``beta`` keeps its reading as that
        band's optical depth.
    """

    sharpness: float = 6.0
    # Reference wavelength (angstrom); ``beta`` is the optical depth *at* it, so this should sit
    # at the bluest band of the filter set in use. Default is the LSST u effective wavelength.
    lam_ref_aa: float = 3671.0

    _prior_sigma = 0.75

    @property
    def _lam_ref_cm(self):
        return self.lam_ref_aa * 1e-8

    def parameter_names(self):
        return ["beta"]

    def parameter_scalings(self):
        return [None]

    def _tau(self, wave_cm, beta):
        # Clip from below: a negative beta (blue boost) with far-UV wavelengths would
        # otherwise overflow exp(-tau); in-band optical values are far from the clip.
        return np.maximum(beta * (self._lam_ref_cm / wave_cm) ** self.sharpness, -50.0)

    def value(self, wave_cm, T, beta):
        return PlanckSpectralTerm.value(wave_cm, T) * np.exp(-self._tau(wave_cm, beta))

    def dvalue_dT(self, wave_cm, T, beta):
        """∂(value)/∂T = ∂Planck/∂T · e^{-τ} (the opacity is T-independent)."""
        return PlanckSpectralTerm.dvalue_dT(wave_cm, T) * np.exp(-self._tau(wave_cm, beta))

    def derivatives(self, wave_cm, T, beta):
        """∂(value)/∂beta = -value·(λ_ref/λ)^sharpness; shape (1, len(wave_cm))."""
        r = (self._lam_ref_cm / wave_cm) ** self.sharpness
        value = PlanckSpectralTerm.value(wave_cm, T) * np.exp(-self._tau(wave_cm, beta))
        jac = np.zeros((1, len(wave_cm)))
        jac[0] = -value * r
        return jac

    def initial_guesses(self, t, m, sigma, band):
        return {"beta": 0.0}

    def limits(self, t, m, sigma, band):
        # beta > ~20 is a saturated u-band blackout; small negative values allow a mild
        # localized blue excess (the analogue of modified_bb's negative beta).
        return {"beta": (-2.0, 20.0)}

    def parameter_priors(self):
        return {"beta": (0.0, self._prior_sigma)}


@dataclass()
class LogParabolaSpectralTerm(BaseSpectralTerm):
    r"""Log-parabola modification of a Planck spectrum.

    .. math::
        F(\lambda) = B_\nu(\lambda, T) \cdot e^{a L + b L^2},\quad L = \ln(\lambda / \lambda_\mathrm{ref})

    Two parameters tilt (``sp_a``) and curve (``sp_b``) the Planck core, both anchored at
    the pure-Planck value 0. This is the most flexible of the deviation terms — its
    curvature captures the *sharpest* blue cutoffs that the single tilt of ``modified_bb``
    cannot — so it gives the best raw fit quality on strongly blanketed sources.

    The cost is that ``(T, sp_a, sp_b)`` over-parameterize the smooth optical SED, so for a
    pure blackbody they are degenerate and ``T`` would be biased. A Gaussian prior anchoring
    ``sp_a`` and ``sp_b`` toward 0 breaks that degeneracy: where the data do not constrain
    the deviation (blackbody-like sources) the prior recovers ``T`` (to ~5-8% with the
    default ``sigma``), while genuinely blanketed sources, which constrain the parameters,
    override it. The prior strength ``_prior_sigma`` tunes the fit-quality vs
    temperature-fidelity trade-off (smaller => more physical T, weaker deviation capture).
    """

    _wave_ref_cm = 6000e-8
    _prior_sigma = 0.5

    @staticmethod
    def parameter_names():
        return ["sp_a", "sp_b"]

    @staticmethod
    def parameter_scalings():
        return [None, None]

    @staticmethod
    def _L_fac(wave_cm, sp_a, sp_b):
        ell = np.log(wave_cm / LogParabolaSpectralTerm._wave_ref_cm)
        return ell, np.exp(sp_a * ell + sp_b * ell * ell)

    @staticmethod
    def value(wave_cm, T, sp_a, sp_b):
        _ell, fac = LogParabolaSpectralTerm._L_fac(wave_cm, sp_a, sp_b)
        return PlanckSpectralTerm.value(wave_cm, T) * fac

    @staticmethod
    def dvalue_dT(wave_cm, T, sp_a, sp_b):
        """∂(value)/∂T = ∂Planck/∂T · exp(aL+bL²)."""
        _ell, fac = LogParabolaSpectralTerm._L_fac(wave_cm, sp_a, sp_b)
        return PlanckSpectralTerm.dvalue_dT(wave_cm, T) * fac

    @staticmethod
    def derivatives(wave_cm, T, sp_a, sp_b):
        """∂(value)/∂(sp_a, sp_b) = value·(L, L²); shape (2, len(wave_cm))."""
        ell, fac = LogParabolaSpectralTerm._L_fac(wave_cm, sp_a, sp_b)
        value = PlanckSpectralTerm.value(wave_cm, T) * fac
        jac = np.zeros((2, len(wave_cm)))
        jac[0] = value * ell
        jac[1] = value * ell * ell
        return jac

    @staticmethod
    def initial_guesses(t, m, sigma, band):
        return {"sp_a": 0.0, "sp_b": 0.0}

    @staticmethod
    def limits(t, m, sigma, band):
        return {"sp_a": (-6.0, 6.0), "sp_b": (-4.0, 4.0)}

    @staticmethod
    def parameter_priors():
        sigma = LogParabolaSpectralTerm._prior_sigma
        return {"sp_a": (0.0, sigma), "sp_b": (0.0, sigma)}


spectral_terms = {
    "planck": PlanckSpectralTerm,
    "blanketed": BlanketedPlanckSpectralTerm,
    # Default (fixed-depth) instance; pass FreeBlanketedPlanckSpectralTerm(free_depth=True)
    # explicitly as the `spectral=` argument for the free-depth variant.
    "free_blanketed": FreeBlanketedPlanckSpectralTerm(),
    "genwien": GenWienSpectralTerm,
    "modified_bb": ModifiedBlackBodySpectralTerm,
    # Default-sharpness instance; pass SharpBlackBodySpectralTerm(sharpness=...) explicitly
    # as the `spectral=` argument to tune the opacity steepness.
    "sharp_bb": SharpBlackBodySpectralTerm(),
    "logparabola": LogParabolaSpectralTerm,
}
