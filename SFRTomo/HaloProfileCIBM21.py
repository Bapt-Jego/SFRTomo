from *.profiles import HaloProfile, HaloProfileNFW
from *.profiles_2pt import Profile2pt
from *.concentration import Concentration
from *.hmfunc import MassFuncTinker10
import numpy as np
from scipy.integrate import simps
from scipy.special import lambertw

class HaloProfileCIBM21(HaloProfile):
    """ CIB profile implementing the model by Maniyar et al.
    (A&A 645, A40 (2021)).

    The parametrization for the mean profile is:

    .. math::
        \\rho_{\\rm SFR}(z) = \\rho_{\\rm SFR}^{\\rm cen}(z)+
        \\rho_{\\rm SFR}^{\\rm sat}(z),

    where the star formation rate (SFR) density from centrals and satellites is
    modelled as:

    .. math::
        \\rho_{\\rm SFR}^{\\rm cen}(z) = \\int_{}^{}\\frac{dn}{dm},
        SFR(M_{\\rm h},z)dm,

    .. math::
        \\rho_{\\rm SFR}^{\\rm sat}(z) = \\int_{}^{}\\frac{dN}{dm},
        (\\int_{M_{\\rm min}}^{M}\\frac{dN_{\\rm sub}}{dm}SFR(M_{\\rm sub},z)dm),
        dm.

    Here, :math:`dN_{\\rm sub}/dm` is the subhalo mass function,
    and the SFR is parametrized as

    .. math::
        SFR(M,z) = \\eta(M,z)\\,
        BAR(M,z),

    where the mass dependence of the efficiency :math:'\\eta' is lognormal

    .. math::
        \\eta(M,z) = \\eta_{\\rm max},
        \\exp\\left[-\\frac{\\log_{10}^2(M/M_{\\rm eff})},
        {2\\sigma_{LM}^2(z)}\\right],
        
    with :math:'\\sigma_{LM}' defined as the redshift dependant logarithmic scatter in mass
    
    .. math::
        \\sigma_{LM}(z) = \\left\\{
        \\begin{array}{cc}
           \\sigma_{LM0} & M < M_{\\rm eff} \\\\
           \\sigma_{LM0} - \\tau max(0,z_{\\rm c}-z) & M \\geq M_{\\rm eff}
        \\end{array}
        \\right.,

    and :math:'BAR' is the Baryon Accretion Rate

    .. math::
        BAR(M,z) = \\frac{\\Omega_{\\rm b}}{\\Omega_{\\rm m}},
        MGR(M,z),

    where :math:'MGR' is the Mass Growth Rate
    
        MGR(M,z) = 46.1\\left(\\frac{M}{10^{12}M_{\\odot}}\\right)^{1.1},
        \\left(1+1.11z\\right)\sqrt{\\Omega_{\\rm m}(1+z)^{3}+\\Omega_{\\rm \\Lambda}}.

    Args:
        cosmo (:obj:`Cosmology`): cosmology object containing
            the cosmological parameters
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        log10meff (float): log10 of the most efficient mass.
        etamax (float) : star formation efficiency of the most efficient mass
        sigLM0 (float): logarithmic scatter in mass.
        tau (float) : rate at which :math:'\\sigma_{LM}' evolves with redshift.
        zc (float) : redshift below which :math:'\\sigma_{LM}' evolves with redshift.
        Mmin (float): minimum subhalo mass.
    """
    name = 'CIBM21'

    def __init__(self, c_M_relation, log10meff=12.7, etamax=0.42,
                 sigLM0=1.75, tau=1.17, zc=1.5, Mmin=1E5):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.l10meff = log10meff
        self.etamax = etamax
        self.sigLM0 = sigLM0
        self.tau = tau
        self.zc = zc
        self.Mmin = Mmin
        self.pNFW = HaloProfileNFW(c_M_relation)
        super(HaloProfileCIBM21, self).__init__()

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent):
        """Subhalo mass function of Tinker & Wetzel (2010ApJ...719...88T)

        Args:
            Msub (float or array_like): sub-halo mass (in solar masses).
            Mparent (float): parent halo mass (in solar masses).

        Returns:
            float or array_like: average number of subhalos.
        """
        return 0.30*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)


    def update_parameters(self, log10meff=None, etamax=None,
                          sigLM0=None, tau=None, zc=None, Mmin=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.

        Args:
            log10meff (float): log10 of the most efficient mass.
            etamax (float) : star formation efficiency of the most efficient mass
            sigLM0 (float): logarithmic scatter in mass.
            tau (float) : rate at which :math:'\\sigma_{LM}' evolves with redshift.
            zc (float) : redshift below which :math:'\\sigma_{LM}' evolves with redshift.
            Mmin (float): minimum subhalo mass (in solar masses).
        """
        if log10meff is not None:
            self.l10meff = log10meff
        if etamax is not None:
            self.etamax = etamax
        if sigLM is not None:
            self.sigLM0 = sigLM0
        if tau is not None:
            self.tau = tau
        if zc is not None:
            self.zc = zc
        if Mmin is not None:
            self.Mmin = Mmin

    def sigLM(self, M, a):
        z = 1/a - 1
        if hasattr(M, "__len__"):
            sig = np.zeros_like(M)
            smallM = np.log10(M) < self.l10meff
            sig[smallM] = self.sigLM0
            sig[~smallM] = self.sigLM0 - self.tau * max(0, self.zc-z)
            return sig
        else:
            if np.log10(M) < self.l10meff:
                return self.sigLM0
            else :
                return self.sigLM0 - self.tau * max(0, self.zc-z)

    def _SFR(self, cosmo, M, a):
        Omega_b = cosmo['Omega_b']
        Omega_m = cosmo['Omega_c'] + cosmo['Omega_b']
        Omega_L = 1 - cosmo['Omega_m']
        z = 1/a - 1
        # Efficiency - eta
        eta = self.etamax * np.exp(-0.5*((np.log(M) - np.log(10)*self.l10meff)/self.sigLM(M, a))**2)
        # Baryonic Accretion Rate - BAR
        MGR = 46.1 * (M/1e12)**1.1 * (1+1.11*z) * np.sqrt(Omega_m*(1+z)**3 + Omega_L)
        BAR = Omega_b/Omega_m * MGR
        return eta * BAR

    def _SFRcen(self, cosmo, M, a):
        fsub = 0.134
        M = M*(1-fsub)
        SFRcen = self._SFR(cosmo, M, a)
        return SFRcen

    def _SFRsat(self, cosmo, M, a):
        fsub = 0.134
        SFRsat = np.zeros_like(M)
        goodM = M >= self.Mmin
        M_use = (1-fsub)*M[goodM, None]
        nm = max(2, 3*int(np.log10(np.max(M_use)/self.Mmin)))
        Msub = np.geomspace(self.Mmin, np.max(M_use), nm+1)[None, :]
        # All these arrays are of shape [nM_parent, nM_sub]

        dnsubdlnm = self.dNsub_dlnM_TinkerWetzel10(Msub, M_use)
        SFRI = self._SFR(cosmo, Msub.flatten(), a)[None, :]
        SFRII = self._SFR(cosmo, M_use, a)*Msub/M_use
        Ismall = SFRI < SFRII
        SFR = SFRI*Ismall + SFRII*(~Ismall)

        integ = dnsubdlnm*SFR*(M_use >= Msub)
        SFRsat[goodM] = simps(integ, x=np.log(Msub))
        return SFRsat
    
    def _real(self, cosmo, r, M, a, mass_def):
        M_use = np.atleast_1d(M)

        SFRs = self._SFRsat(cosmo, M_use, a)
        ur = 1

        prof = SFRs[:, None]*ur
        
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)

        SFRc = self._SFRcen(M_use, a)
        SFRs = self._SFRsat(M_use, a)
        uk = 1
        prof = SFRc[:, None]+SFRs[:, None]*uk
        
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        
        SFRc = self._SFRcen(M_use, a)
        SFRs = self._SFRsat(M_use, a)
        uk = 1

        prof = SFRs[:, None]*uk
        prof = 2*SFRc[:, None]*prof + prof**2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class Profile2ptCIBM21(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the CIB profile. It follows closely the
    implementation of the equivalent HOD quantity
    (see :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`
    and Eq. 15 of McCarthy & Madhavacheril (2021PhRvD.103j3515M)).
    """
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment for the CIB
        profile.

        Args:
            prof (:class:`HaloProfileCIBM21`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            prof2 (:class:`HaloProfileCIBM21`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation. Note that only auto-correlations
                are allowed in this case.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.

        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if not isinstance(prof, HaloProfileCIBM21):
            raise TypeError("prof must be of type `HaloProfileCIB`")
        if prof2 is not None:
            if not isinstance(prof2, HaloProfileCIBM21):
                raise TypeError("prof must be of type `HaloProfileCIB`")
        return prof._fourier_variance(cosmo, k, M, a, mass_def)
