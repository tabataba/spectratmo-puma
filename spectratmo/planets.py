"""Planets
==========

.. autoclass:: Planet
   :members:
   :private-members:

"""

from math import pi



class Planet(object):
    """Represent a planet.

    Parameters
    ----------

    name : str

      Name of the planet.

    radius : number

      Radius (meters).

    Omega : number

      Rotation frequency (Hz).

    pressure_m : number

      Mean pressure at sea level (hPa).

    """
    def __init__(self, name, radius=1., Omega=1., pressure_m=1.):
        self.name = name
        self.radius = float(radius)
        self.Omega = float(Omega)
        self.pressure_m = float(pressure_m)
        #if name == 'Earth': self=earth
        #if name == 'Mars' : self=mars
        if name == 'Earth':
            self.name=name
            self.radius=6367470.  # Earth radius (meters)
            self.Omega=2*pi/86164  # Rotation frequency (Hz)
            self.pressure_m=101300.  # Mean pressure at sea level (Pa)
        
        if name == 'Mars' :
            self.name=name
            self.radius=3389500.  # Mars radius (meters)
            self.Omega=2*pi/88800  # Rotation frequency (Hz)
            self.pressure_m=630  # Mean pressure at sea level (Pa)

        #mars = Planet('Mars', radius=3389500., Omega=2*pi/88800, pressure_m=6.3)

        #if name == 'Earth': self=earth
        #if name == 'Mars' : self=mars

earth = Planet('Earth')

mars = Planet('Mars')
