import pandas as pd
import math
import numpy as np

class Basepath(object):
    basepath = None

    def __int__(self):
        pass


    def load_csv(self, path, x='E', y='N', sep='\t'):
        """Load csv"""
        self.basepath = pd.read_csv(path, sep=sep)
        try:
            self.basepath['E'] = self.basepath[x]
            self.basepath['N'] = self.basepath[y]
        except ValueError:
            print "Oops!  The provided (x,y) column names do not exist.  Try again..."

    def calculate_s(self):
        self.basepath['Dist'] = np.nan
        try:
            sum1 = 0
            for i in range(1, len(self.basepath)):
                sum1 += self.disti(self.basepath.E[i - 1], self.basepath.N[i - 1],self.basepath.E[i], self.basepath.N[i])
                print(sum1)
                self.basepath['Dist'][i] = float(sum1)
                print(u'Intervalo medio entre pontos de tracado:')
                print(sum1 / len(df))
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
            self.basepath['bc'][1:] = map(lambda x: self.disti(self.basepath.E[x-1],self.basepath.N[x-1],
                                                           self.basepath.E[x],self.basepath.N[x]), self.basepath.index[1:])
        except RuntimeError:
            print('Runtime error is raised')
            raise


    def calculate_vbc(self):
        """ 
        :return: calculated vbc
        """
        self.basepath['vbc'] = np.nan
        try:
            self.basepath['vbc'][1:] = pd.Series( map(lambda x:
                                    np.array([self.basepath.E[x] - self.basepath.E[x - 1],
                                            self.basepath.N[x] - self.basepath.N[x - 1]]) / self.basepath['bc'][x],
                                            self.basepath.index[1:]))
        except RuntimeError:
            print('Runtime error is raised')
            raise
