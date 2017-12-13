# -*- coding: utf-8 -*-
"""
Basegrid

Defines the Basegrid object to be used by the class Transformation.

"""

import pandas as pd

#pylint: disable=no-name-in-module
class Basegrid(object):
    """Basegrid object can read from csv, shapefile or Qgis Layer.

    It can only be diretly saved as a layer,
    however, basegrid is a pandas DataFrame object and can be saved as such.

    Args:
        basegrid (DataFrame): Pandas DataFrame, can be loaded with the available
                            methods or directly. 'E' and 'N' are the columns
                            that should hold the coordinates.

    """

    def __init__(self):

        self.basegrid = pd.DataFrame()
        self.basegrid_layer = None

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
            self.basegrid['E'] = zip(*coordinates)[0]
            self.basegrid['N'] = zip(*coordinates)[1]
        except RuntimeError:
            print 'Runtime error is raised'
            raise


    def load_csv(self, path, x='X', y='Y', sep='\t'):
        """Load csv"""

        self.basegrid = pd.read_csv(path, sep=sep)
        try:
            self.basegrid['E'] = self.basegrid[x]
            self.basegrid['N'] = self.basegrid[y]
        except ValueError:
            print """Oops! The provided (x,y) column names do not exist,
                    or were not give."""

    def load_shp(self, path):
        """Load shp

        Args:
            basegrid (GeoDataFrame): GeoPandas GepDataFrame for the basegrid"""
        try:

            import geopandas as gpd

        except RuntimeError:

            print 'Need geopandas for that. Try: pip install geopandas'
            raise

        self.basegrid = gpd.GeoDataFrame()
        self.basegrid = gpd.read_file(path)

        try:
            self.basegrid.loc[:,
                              ('E')] = [self.basegrid.geometry.get_values()[p].x
                                        for p in  range(len(self.basegrid))]
            self.basegrid.loc[:,
                              ('N')] = [self.basegrid.geometry.get_values()[p].y
                                        for p in  range(len(self.basegrid))]

        except ValueError:

            print """Oops!  The provided file does not have a proper (x,y)
                     geometry column.  Try again..."""

    def save_layer(self, layer, output=None):
        """Save Layer - Used in Qgis Plugin

        Args:
            inLayerGeometryType (str): Geometry type from original basegrid to
                                     be saved to the new transformed one.
            inFields (array): List of fields the original basegrid layer have.
            inFeatures (array): List of features lists.
            inLayerCRS (str): The original basegrid layer projection code.
        """

        try:
            from qgis.core import (QgsVectorFileWriter,
                                   QgsVectorLayer,
                                   QgsField, QgsPoint,
                                   QgsMapLayerRegistry,
                                   QgsGeometry,
                                   QgsFeature)

            from PyQt4.QtCore import QVariant

        except RuntimeError:

            print 'Need qgis.core for that.'
            raise

        # basegrid layer information

        inLayerGeometryType = ['Point', 'Line', 'Polygon'][layer.geometryType()]
        inFields = layer.dataProvider().fields()
        inFeatures = [feat for feat in layer.getFeatures()]
        inLayerCRS = layer.crs().authid()

        # new basegrid_layer creation

        self.basegrid_layer = QgsVectorLayer(inLayerGeometryType
                                             + '?crs='+inLayerCRS,
                                             layer.name()
                                             + u'_new', 'memory')
        self.basegrid_layer.startEditing()

        # setting dataprovider and adding new fields s and d

        provider = self.basegrid_layer.dataProvider()
        provider.addAttributes(inFields.toList())

        provider.addAttributes([
            QgsField("s", QVariant.Double),
            QgsField("d", QVariant.Double)])

        self.basegrid_layer.commitChanges()
        QgsMapLayerRegistry.instance().addMapLayer(self.basegrid_layer)

        # loading with attribute values

        fields = provider.fields()
        features = []
        i = 0
        for feat in inFeatures:
            point = QgsPoint(self.basegrid['E'][i], self.basegrid['N'][i])
            geometry = QgsGeometry.fromPoint(point)
            feature = QgsFeature()
            feature.setGeometry(geometry)
            feature.setFields(fields)
            # BUG - Add type verification, if = ['0'] set as none
            feature.setAttributes(feat.attributes()
                                  + [float(self.basegrid['s'][i])]
                                  +[float(self.basegrid['d'][i])])

            features.append(feature)
            i += 1
        provider.addFeatures(features)
        print output
        if output != None:
            QgsVectorFileWriter.writeAsVectorFormat(self.basegrid_layer,
                                                    output, "utf-8",
                                                    None, "ESRI Shapefile")

        self.basegrid_layer.commitChanges()
        self.basegrid_layer.updateExtents()
