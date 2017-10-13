import pandas as pd
import math
import numpy as np
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt


class Transformation(object):

    def __int__(self, basepath, basegrid):

        self.basepath = basepath.basepath
        self.basegrid = basegrid.basegrid
        self.bp = None
        self.vbp = None
        self.pbc = None
        self.pto = None

    def calculate_bp(self):
        """ 
        :param self: bp (distance bp)
        :return: calculate distance matrix from basepath to basegrid
        """
        self.bp = cdist(self.basepath[['E','N']].as_matrix(), self.basegrid[['E','N']].as_matrix())

    def calculate_vbp(self):
        """ 
        :param self: vbp (vector bp)
        :return: None
        """

        try:
            a = np.stack([self.basepath[['E', 'N']].as_matrix()]*len(self.basegrid))
            b = np.stack([self.basegrid[['E', 'N']].as_matrix()]*len(self.basepath)).reshape(a.shape)

            # a little magic
            c = pd.DataFrame(np.vstack(a-b))
            c[0] = c[0]/ np.hstack(bp)
            c[1] = c[1]/ np.hstack(bp)

            self.vbp = c.as_matrix().reshape(np.stack(a-b).shape)

        except RuntimeError:
            print('Runtime error is raised')
            raise

    def calculate_pbc(self):

        try:
            vbc = self.basepath.vbc.as_matrix()
            pbc = map(lambda x: cdist(np.matrix(vbc[x]), self.vbp[x],lambda u,v: np.arccos(np.dot(u,v))), range(1,len(vbc)))
            self.pbc = np.stack(pbc)
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def calculate_pto(self):
        try:
            vbc = self.basepath.vbc.as_matrix()
            mtp1 = vbc * np.cos(self.pbc).T
            mtp2 = np.multiply(self.bp[:-1].T,mtp1)
            coord = self.basepath[['E','N']].as_matrix()
            self.pto = map(lambda x: coord[:-1][x]+ list(mtp2.T[x]),range(len(coord[:-1])))
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def vmod(self):

        return np.sqrt((x*x).sum(axis=1))

    def set_index_correlation(self):

        try:
            coord = self.basepath[['E', 'N']].as_matrix()
            # Distancia Pto - B
            cd1 = map(lambda i: np.array(self.vmod(coord[:-1][i]- self.pto[i])), range(len(coord[:-1])))
            # Distancia Pto - C
            cd2 = map(lambda i: np.array(self.vmod(coord[1:][i] - self.pto[i])), range(len(coord[1:])))
            # Distancia C - B
            cd3 = self.vmod(coord[1:] - coord[:-1])
            # Refatorando o vetor cd3 para o tamanho de cd4
            cd3_2 = np.stack([cd3] * len(self.basegrid))
            # Soma de cd1 e cd2. (Deve ser igual a distancia cd3)
            cd4 = np.array(cd1) + np.array(cd2)
            # cd4 - cd3 (soma cd1 cd2)
            cd5 = pd.DataFrame(cd4 - cd3_2.T)

            self.basegrid['id_basepath'] = cd5[cd5<0.00000001].idxmin()
            self.basegrid['id_basepath'][self.basegrid['id_basepath'].isnull()] = 0
            self.basegrid['id_basepath'] = self.basegrid['id_basepath'].astype(int)

        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_pto(self):

        try:
            self.basegrid['ptoE'] = np.nan
            self.basegrid['ptoN'] = np.nan

            self.basegrid[['ptoE', 'ptoN']] = np.array(self.pto)[self.basegrid['id_basepath'], range(len(self.basegrid['id_basepath']))]
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_position(self):
        try:
            bE = np.stack([self.basepath.E[:-1]] * len(self.basegrid)).T
            bN = np.stack([self.basepath.N[:-1]] * len(self.basegrid)).T
            cE = np.stack([self.basepath.E[1:]] * len(self.basegrid)).T
            cN = np.stack([self.basepath.N[1:]] * len(self.basegrid)).T
            pE = np.stack([self.basegrid.X] * len(cE))
            pN = np.stack([self.basegrid.Y] * len(cE))
            pst = np.sign(np.multiply(cE - bE, pN - bN) - np.multiply(cN - bN, pE - bE))
            self.basegrid.position = pst
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_d(self):
        try:
            d = np.array(self.bp)[self.basegrid['id_basepath'],
                                  range(len(self.basegrid['id_basepath']))] * np.sin(np.array(self.pbc)[self.basegrid['id_basepath'],
                                                                                                        range(len(self.basegrid['id_basepath']))])
            self.basegrid['d'] = d * self.basegrid.position
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_s(self):
        try:
            s_increment = np.array(self.bp)[self.basegrid['id_basepath'], range(len(self.basegrid))] * np.cos(
                np.array(self.pbc)[self.basegrid['id_basepath'], range(len(self.basegrid))])
            s_matrix = np.stack([self.basepath.Dist] * len(self.basegrid))
            s_b = np.array(s_matrix.T)[self.basegrid['id_basepath'], range(len(self.basegrid))]

            self.basegrid['sincrement'] = s_increment
            self.basegrid['sB'] = s_b
            self.basegrid['s'] = s_b + s_increment
        except RuntimeError:
            print('Runtime error is raised')
            raise


    def plot_s_d(self):

        plt.figure(figsize=(20, 20))
        plt.scatter(self.basegrid.s[self.basegrid.s.notnull()], self.basegrid.d[self.basegrid.s.notnull()])
        plt.show()

    def run(self):

        self.calculate_bp()
        self.calculate_vbp()
        self.calculate_pbc()
        self.calculate_pto()
        self.set_index_correlation()
        self.set_pto()
        self.set_position()
        self.set_d()
        self.set_s()
        self.plot_s_d()







