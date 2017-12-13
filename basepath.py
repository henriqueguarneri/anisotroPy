# -*- coding: utf-8 -*-
"""
Basepath

Defines the Basepath object to be used by the class Transformation.

"""

import math
import pandas as pd
import numpy as np

# pylint: disable=no-member
class Basepath(object):
    """Basepath object can read from csv, shapefile or Qgis Layer.

    It can't be diretly,
    however, basepathis a pandas DataFrame object and can be saved as such.

    Args:
        basepath (DataFrame): Pandas DataFrame, can be loaded with the available
                            methods or directly. 'E' and 'N' are the columns
                            that should hold the coordinates.

    """

    def __init__(self):

        self.basepath = pd.DataFrame()

    def load_layer(self, layer):
        """Load qgis layer

        Args:
            coordinates (array): Used to load geometry features from layer.
        Todo:
            include test to check if input has metric coordinates.
        """

        try:
            coordinates = []
            for f in layer.getFeatures():
                coordinates.append(f.geometry().asPoint())
            self.basepath['E'] = zip(*coordinates)[0]
            self.basepath['N'] = zip(*coordinates)[1]
        except RuntimeError:
            print 'Runtime error is raised'
            raise

    def load_csv(self, path, x='E', y='N', sep='\t'):
        """Load csv"""
        self.basepath = pd.read_csv(path, sep=sep)
        try:
            self.basepath['E'] = self.basepath[x]
            self.basepath['N'] = self.basepath[y]
        except ValueError:
            print """Oops! The provided (x,y) column names do not exist,
                    or were not give."""

    def load_shp(self, path):
        """Load shp"""
        try:
            import geopandas as gpd
        except RuntimeError:
            print 'Need geopandas for that. Try: pip install geopandas'
            raise
        self.basepath = gpd.GeoDataFrame()
        self.basepath = gpd.read_file(path)
        try:
            self.basepath.loc[:,
                              ('E')] = [self.basepath.geometry.get_values()[p].x
                                        for p in  range(len(self.basepath))]
            self.basepath.loc[:,
                              ('N')] = [self.basepath.geometry.get_values()[p].y
                                        for p in  range(len(self.basepath))]
        except ValueError:
            print """Oops!  The provided file does not have a proper (x,y)
                     geometry column.  Try again..."""

    def calculate_s(self):
        """ calculate_s: Calculate the cumulative distance between the points

            Args:

                sum1 (int): Auxiliary variable to define the S(Dist) coordinate.
        """
        self.basepath['Dist'] = 0.
        try:
            sum1 = 0
            for i in range(1, len(self.basepath)):
                sum1 += self.disti(self.basepath.E[i - 1],
                                   self.basepath.N[i - 1],
                                   self.basepath.E[i],
                                   self.basepath.N[i])

                self.basepath.loc[i:i, ('Dist')] = float(sum1)

            print u'Avarage interval between path points:'
            print sum1 / len(self.basepath)
        except RuntimeError:
            print 'Runtime error is raised'
            raise

    @classmethod
    def disti(cls, x0, y0, x1, y1):
        """ Classic euclidian distance
            Args:
                distance (float): Distance between two points.
                x0 (flot): First metric coordinate of a x point.
                y0 (flot): First metric coordinate of a y point.
                x1 (flot): First metric coordinate of a x point.
                y1 (flot): First metric coordinate of a y point.
        """

        distance = math.pow(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2), 0.5)
        return distance

    def calculate_bc(self):
        """ Calculate the non-cumulative distance between
            the consecutive set of points."""

        self.basepath['bc'] = np.nan
        try:
            self.basepath.loc[1:,
                              ('bc')] = [self.disti(self.basepath.E[x-1],
                                                    self.basepath.N[x-1],
                                                    self.basepath.E[x],
                                                    self.basepath.N[x])
                                         for x in self.basepath.index[1:]]

            print 'bc calculated successfully'
        except RuntimeError:
            print 'Runtime error is raised'
            raise

    def calculate_vbc(self):
        """ Define the vector between each consecutive point
        Args:
            vbc (pandas.Series): series with the vectors between
                                 the set of consecutive points.
        """
        self.basepath['vbc'] = np.nan
        try:
            vbc = pd.Series(
                [np.array([self.basepath.E[x] - self.basepath.E[x - 1],
                           self.basepath.N[x] - self.basepath.N[x - 1]])
                 / self.basepath['bc'][x] for x in self.basepath.index[1:]])

            vbc.index += 1
            self.basepath.loc[1:, ('vbc')] = vbc
            print 'vbc calculated successfully'
        except RuntimeError:
            print 'Runtime error is raised'
            raise
