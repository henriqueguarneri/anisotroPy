import pandas as pd
import math
import numpy as np

class Basepath(object):
    basepath = None

    def __int__(self):
        pass

    def load_layer(self, layer):

        # include test to check if metric coordinates

        try:
            from qgis.core import *
        except RuntimeError:
            print('Need qgis.core for that.')
            raise

        self.basepath = pd.DataFrame()

        try:
            coordinates = []
            for f in layer.getFeatures():
                coordinates.append(f.geometry().asPoint())
            self.basepath['E'] = zip(*coordinates)[0]
            self.basepath['N'] = zip(*coordinates)[1]
        except RuntimeError:
            print('Runtime error is raised')
            raise   

    def load_csv(self, path, x='E', y='N', sep='\t'):
        """Load csv"""
        self.basepath = pd.read_csv(path, sep=sep)
        try:
            self.basepath['E'] = self.basepath[x]
            self.basepath['N'] = self.basepath[y]
        except ValueError:
            print "Oops!  The provided (x,y) column names do not exist.  Try again..."

    def load_shp(self, path):
        """Load shp"""
        try:
            import geopandas as gpd
        except RuntimeError:
            print('Need geopandas for that. Try: pip install geopandas')
            raise

        self.basepath = gpd.read_file(path)
        try:
            self.basepath.loc[:,('E')] = map(lambda p: self.basepath.geometry.get_values()[p].x, range(len(self.basepath)))
            self.basepath.loc[:,('N')] = map(lambda p:self.basepath.geometry.get_values()[p].y, range(len(self.basepath)))
        except ValueError:
            print "Oops!  The provided file does not have a proper (x,y) geometry column.  Try again..."

    def calculate_s(self):
        self.basepath['Dist'] = 0.
        try:
            sum1 = 0
            for i in range(1, len(self.basepath)):
                sum1 += self.disti(self.basepath.E[i - 1], self.basepath.N[i - 1],self.basepath.E[i], self.basepath.N[i])
                #print(sum1)
                self.basepath.loc[i:i,('Dist')] = float(sum1)
            print(u'Avarage interval between path points:')
            print(sum1 / len(self.basepath))
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def disti(self, x0, y0, x1, y1):
        """ Distancia euclidiana entre dois pontos"""
        r = math.pow(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2), 0.5)
        return r

    def calculate_bc(self):
        self.basepath['bc'] = np.nan
        try:
            self.basepath.loc[1:,('bc')] = map(lambda x: self.disti(self.basepath.E[x-1],self.basepath.N[x-1],
                                                           self.basepath.E[x],self.basepath.N[x]), self.basepath.index[1:])
            print('bc calculated successfully')
        except RuntimeError:
            print('Runtime error is raised')
            raise


    def calculate_vbc(self):
        """ 
        :return: calculated vbc
        """
        self.basepath['vbc'] = np.nan
        try:
            vbc = pd.Series( map(lambda x:
                                    np.array([self.basepath.E[x] - self.basepath.E[x - 1],
                                            self.basepath.N[x] - self.basepath.N[x - 1]]) / self.basepath['bc'][x],
                                            self.basepath.index[1:]))
            vbc.index += 1
            self.basepath.loc[1:,('vbc')] = vbc
            print('vbc calculated successfully')
        except RuntimeError:
            print('Runtime error is raised')
            raise
