"""Spatial representation
=========================


"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


from spectratmo.output import BaseOutput


class SpatialRepr(BaseOutput):

    _name = 'spatial_repr'

    def plot(self, key, instant, ip, num_fig=None, quiver=True):
        ds = self._dataset

        if quiver:
            u = ds.get_spatial2dvar('u', instant, ip)
            v = ds.get_spatial2dvar('v', instant, ip)

            if key == 'u':
                data = u
            elif key == 'v':
                data = v
            else:
                data = ds.get_spatial2dvar(key, instant, ip)
        else:
            data = ds.get_spatial2dvar(key, instant, ip)

        plt.figure(num=num_fig)
        ax = plt.axes(
            # projection=ccrs.AzimuthalEquidistant()
            # projection=ccrs.PlateCarree()
            projection=ccrs.Robinson()
            # projection=ccrs.Mercator()
        )

        # make the map global rather than have it zoom in to
        # the extents of any plotted data
        ax.set_global()

        if ds.planet.name == 'Earth':
            ax.coastlines()

        lats = np.rad2deg(np.linspace(-np.pi / 2, np.pi / 2, ds.nlat))
        lons = np.rad2deg(np.linspace(0, 2 * np.pi, ds.nlon))
        lons, lats = np.meshgrid(lons, lats)

        ax.contourf(lons, lats, data,
                    transform=ccrs.PlateCarree(),
                    cmap='spectral')

        if quiver:
            ax.quiver(lons, lats, u, v, transform=ccrs.PlateCarree(),
                      regrid_shape=30)

        # ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
        # ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        plt.show()


if __name__ == '__main__':

    from spectratmo.datasets.puma import PUMADataSet

    ds = PUMADataSet()

    ds.spatial_repr.plot('t', 1, 1)
