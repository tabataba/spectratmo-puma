"""Global time averages
=======================

.. autoclass:: GlobalTMeanBase
   :members:
   :private-members:

.. autoclass:: GlobalTMeanWithBeta
   :members:
   :private-members:

.. autoclass:: GlobalTMeanWithoutBeta
   :members:
   :private-members:

"""

from spectratmo.output import BaseOutput

from spectratmo.phys_const import R,khi

from spectratmo.planets import earth

import numpy as np

class GlobalTMeanBase(BaseOutput):
    """Handle the basic quantities obtained by global and time averaging."""
    _name = 'global_tmean'


    def plot_gamma(self):
        raise NotImplementedError


class GlobalTMeanWithBeta(GlobalTMeanBase):
    r"""Handle the basic quantities obtained by global and time averaging.

    .. |p| mathmacro:: \partial

    - Time-average surface pressure :math:`\bar{p_s}(x_h)`

    - smoothed function :math:`\beta(p, x_h)`

      .. math::

         \beta(p, x_h) = smooth(H(\bar{p_s}(x_h) - p))

    - horizontal mean of :math:`\beta(p, x_h)`

      .. math::

         \beta(p) = \langle \beta(p, x_h) \rangle

    - representative horizontal mean :math:`\langle \theta(p, x_h)
      \rangle_r (p) = \langle \beta \theta \rangle/\langle\beta\rangle`

    - function :math:`\gamma(p)` used in the computation of the APE:

      .. math::

         \gamma(p) = \frac{R}{- \Gamma(p) p \p_p \langle\theta\rangle_r}

    - maybe also the ratio

      .. math::

         \frac{\p_p \gamma}{\gamma}

    """

    with_beta = True

    def compute_tmean_ps(self):
        r"""Compute the time average of the surface pressure.

        """
        ds = self._dataset
        ps = ds.oper.create_array_spat(0.)
        for instant in ds.instants:
            ps += ds.get_spatial2dvar('ps', instant)
        ps /= ds.nb_instants
        return ps

    def compute_beta(self):
        r"""Compute beta from the time-averaged surface pressure.

        Notes
        -----

        .. math::

           \beta(p, x_h) = H(\bar{p_s}(x_h) - p)

        """
        ds = self._dataset
        ps = np.reshape(self.compute_tmean_ps(), (1, ds.nlat, ds.nlon))
        pl = np.reshape(ds.pressure_levels, (ds.nlev, 1, 1))
        beta = ds.create_array_spat3d(0.)
        beta[(ps-pl) >= 0] = 1.0
        return beta

    def plot_tmean_ps(self):
        raise NotImplementedError

    def compute_gamma(self):
        r"""Compute the mean temperature and gamma.

        Notes
        -----

        .. math::

           \gamma(p) = \frac{R}{- \Gamma(p) p \partial_p \langle
           \theta \rangle_r}

        """
        raise NotImplementedError


class GlobalTMeanWithoutBeta(GlobalTMeanBase):
    r"""Handle the basic quantities obtained by global and time averaging.

    .. |p| mathmacro:: \partial

    - representative horizontal mean :math:`\langle \theta(p, x_h)
      \rangle_r (p) = \langle \theta \rangle`

    - function :math:`\gamma(p)` used in the computation of the APE:

      .. math::

         \gamma(p) = \frac{R}{- \Gamma(p) p \p_p \langle\theta\rangle_r}

    - maybe also the ratio

      .. math::

         \frac{\p_p \gamma}{\gamma}

    """
    with_beta = False

    def compute_gamma(self):
        r"""Compute the mean temperature and gamma.

        Notes
        -----

        .. math::

           \gamma(p) = \frac{R}{- \Gamma(p) p \partial_p \langle
           \theta \rangle}

        """
        ds = self._dataset
        if self.check_if_file_exists('gamma'):
            gamma = self.get_spat1d('gamma')
            #print 'gamma found'
        else:
            #print 'gamma not found'
            dp = np.zeros(ds.nlev)
            dp[0] = ds.pressure_levels[0] - ds.pressure_levels[1]
            dp[ds.nlev-1] = (
                ds.pressure_levels[ds.nlev-2] - ds.pressure_levels[ds.nlev-1])
            for ip in range(1, ds.nlev-1):
                dp[ip] = ((ds.pressure_levels[ip-1] +
                           ds.pressure_levels[ip])/2.0 -
                          (ds.pressure_levels[ip] +
                           ds.pressure_levels[ip+1])/2.0)
            tm = np.zeros(ds.nlev)
            for ip in range(ds.nlev):
                thetamean = 0#ds.oper.create_array_spat(0.)
                for instant in ds.instants:
                    #print instant
                    thetamean += (
                        np.mean(ds.get_spatial2dvar('t', instant, ip) *
                        (earth.pressure_m/ds.pressure_levels[ip])**khi))
                thetamean /= ds.nb_instants
                tm[ip]=thetamean

            dtmdp = np.zeros(ds.nlev)
            dtmdp[0] = (tm[0] - tm[1]) / dp[0]
            dtmdp[ds.nlev-1] = (tm[ds.nlev-2] - tm[ds.nlev-1]) / dp[ds.nlev-1]
            for ip in range(1,ds.nlev-1):
                dtmdp[ip] = (tm[ip-1] - tm[ip+1]) / (2.0* dp[ip])

            gamma = np.zeros(ds.nlev)
            for ip in range(ds.nlev):
                gamma[ip] = R / (-(earth.pressure_m/ds.pressure_levels[ip])**khi
                                 * ds.pressure_levels[ip] * dtmdp[ip])
            self.save_spat1d(gamma, 'gamma')
        return gamma
