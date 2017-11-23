import pandas as pd
import math
import numpy as np


class Basegrid(object):
    basegrid = None

    def __int__(self):
        pass

    def load_layer(self, layer):
        """Load qgis layer"""
        # include test to check if metric coordinates

        try:
            from qgis.core import *
        except RuntimeError:
            print('Need qgis.core for that.')
            raise

        self.basegrid = pd.DataFrame()

        try:
            coordinates = []
            for f in layer.getFeatures():
                coordinates.append(f.geometry().asPoint())
            self.basegrid['E'] = zip(*coordinates)[0]
            self.basegrid['N'] = zip(*coordinates)[1]
        except RuntimeError:
            print('Runtime error is raised')
            raise


    def load_csv(self, path, x='X', y='Y', sep='\t'):
        """Load csv"""
        self.basegrid = pd.read_csv(path, sep=sep)
        try:
            self.basegrid['E'] = self.basegrid[x]
            self.basegrid['N'] = self.basegrid[y]
        except ValueError:
            print "Oops!  The provided (x,y) column names do not exist.  Try again..."

    def load_shp(self, path):
        """Load shp"""
        try:
            import geopandas as gpd
        except RuntimeError:
            print('Need geopandas for that. Try: pip install geopandas')
            raise

        self.basegrid = gpd.read_file(path)
        try:
            self.basegrid.loc[:,('E')] = map(lambda p: self.basegrid.geometry.get_values()[p].x, range(len(self.basegrid)))
            self.basegrid.loc[:,('N')] = map(lambda p:self.basegrid.geometry.get_values()[p].y, range(len(self.basegrid)))
        except ValueError:
            print "Oops!  The provided file does not have a proper (x,y) geometry column.  Try again..."

    def save_layer(self, layer, output=None):
        """Save Layer - Used in Qgis Plugin"""
        try:
            from qgis.core import *
            from PyQt4.QtCore import QVariant   
        except RuntimeError:
            print('Need qgis.core for that.')
            raise

        # basegrid layer information
        
        inLayerGeometryType = ['Point','Line','Polygon'][layer.geometryType()]
        inFields = layer.dataProvider().fields()
        inFeatures = [ feat for feat in layer.getFeatures() ]
        inLayerCRS = layer.crs().authid()

        # new basegrid_layer creation
        self.basegrid_layer = QgsVectorLayer(inLayerGeometryType + '?crs='+inLayerCRS, layer.name() + u'_new', 'memory')#"point?crs=epsg:32722&field=s:double&field=d:double&index=yes", "basegrid", "memory")
        self.basegrid_layer.startEditing()

        # setting dataprovider and adding new fields s and d
        
        provider = self.basegrid_layer.dataProvider()
        provider.addAttributes(inFields.toList())
        provider.addAttributes([QgsField("s",  QVariant.Double), QgsField("d", QVariant.Double)])
        self.basegrid_layer.commitChanges()
        QgsMapLayerRegistry.instance().addMapLayer(self.basegrid_layer)
        
        # loading with attribute values
        
        fields = provider.fields()
        features = []
        i = 0
        for feat in inFeatures:
            point = QgsPoint(self.basegrid['E'][i],self.basegrid['N'][i])
            geometry = QgsGeometry.fromPoint(point)
            feature = QgsFeature()
            feature.setGeometry(geometry)
            feature.setFields(fields)
            # BUG - Add type verification, if = ['0'] set as none
            feature.setAttributes(feat.attributes() + [float(self.basegrid['s'][i])]+[float(self.basegrid['d'][i])])
            features.append(feature)            #feature.setAttribute("s", float(self.basegrid['s'][i]))
            i+=1
        provider.addFeatures(features)
        print output
        if output != None:
            _writer = QgsVectorFileWriter.writeAsVectorFormat(self.basegrid_layer,output,"utf-8",None,"ESRI Shapefile")


        self.basegrid_layer.commitChanges()
        self.basegrid_layer.updateExtents()
