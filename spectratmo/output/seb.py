"""Spectral energy budget
=========================


"""

from spectratmo.output import BaseOutput


class SpectralEnergyBudget(BaseOutput):

    _name = 'seb'

    def compute_1instant(self):
        r"""Compute spectral energy budget

        Notes
        -----

        .. math::

           E_K(l) =

           E_A(l) =

        """
        raise NotImplementedError

    def compute_loopallinstants(self):
        raise NotImplementedError

    def compute_tmean(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
