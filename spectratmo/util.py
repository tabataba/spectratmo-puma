"""Utilities
============

.. autoclass:: SetOfVariables
   :members:
   :private-members:

.. autoclass:: SetOfSpectra
   :members:
   :private-members:

.. autoclass:: StackOfSetOfVariables
   :members:
   :private-members:

"""

import numpy as np

from spectratmo.phys_const import g, Gamma_dah


class SetOfVariables(object):
    """Set of variables on one pressure level."""

    def __init__(self, name_type_variables):
        self.name_type_variables = name_type_variables
        self.ddata = dict()

    def __add__(self, other):
        if isinstance(other, SetOfVariables):
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]+other.ddata[k]
        elif isinstance(other, int) or isinstance(other, float):
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]+other
        else:
            raise ValueError()

        return obj_result
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, SetOfVariables):
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]-other.ddata[k]
        else:
            raise ValueError()

        return obj_result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = other*self.ddata[k]
        else:
            raise ValueError()
        return obj_result

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, (int, float)):
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]/other
        else:
            raise ValueError()
        return obj_result

    def reset_to_zeros(self):
        for key, item_data in self.ddata.iteritems():
            if np.isscalar(item_data):
                self.ddata[key] = 0.
            else:
                item_data[:] = 0.


class SetOfSpectra(SetOfVariables):
    """Set of energy spectra."""
    def add_APE_spectrum(self, Tmean, GammaMean):
        self.ddata['E_APE_n'] = (self.ddata['E_TT_n'] * g /
                                 (Tmean*(Gamma_dah-GammaMean)))
        self.ddata['E_APEb_n'] = (self.ddata['E_TTb_n'] *
                                  g / (Tmean*(Gamma_dah-GammaMean)))





class StackOfSetOfVariables(object):
    """..."""
