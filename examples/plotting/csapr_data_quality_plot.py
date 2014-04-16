#!/usr/bin/env python
#first we do some imports and check the version of Py-ART for consistency
from matplotlib import use
use('agg')
import pyart
from matplotlib import pyplot as plt
import netCDF4
import numpy as np
from copy import deepcopy
import sys
print pyart.__version__

def nice_pair(radar, fields_to_plot, ranges):
    display = pyart.graph.RadarMapDisplay(radar)
    nplots = len(fields_to_plot)
    plt.figure(figsize=[7 * nplots, 4])
    for plot_num in xrange(nplots):
        field = fields_to_plot[plot_num]
        vmin, vmax = ranges[plot_num]
        plt.subplot(1, nplots, plot_num + 1)
        display.plot_ppi_map(field, 0, vmin=vmin,
                             vmax=vmax, resolution = 'l')
        display.plot_range_rings([50,100])

if __name__=="__main__":
  datafile = sys.argv[1]
  radar = pyart.io.read(datafile)
  tstr = pyart.io.common.netCDF4.num2date(radar.time['data'],
            units=radar.time['units'])[0].strftime('%Y%m%d%H%M%S')
  fname_out = sys.argv[2] + radar.metadata['instrument_name'] + '_PPI_' \
              + tstr + '.png'
  print fname_out
  nice_pair(radar,
          ['clutter_filtered_reflectivity','clutter_filtered_copolar_correlation_coefficient' ],
          [(-8,16), (.8, 1.0)])
  plt.savefig(fname_out)
