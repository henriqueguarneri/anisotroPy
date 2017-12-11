import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import gc



class Transformation(object):

    def __init__(self, basepath, basegrid, msp=1000, st = 0.0001):

        self.basepath = basepath.basepath
        self.basegrid = basegrid.basegrid
        self.maximum_search_parameter = msp
        self.search_tolerance = st
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

    def calculate_vbp_old(self):
        
        vbpX = cdist(self.basepath[['E']].as_matrix(), self.basegrid[['E']].as_matrix(),lambda u, v: u-v)/self.bp
        vbpY = cdist(self.basepath[['N']].as_matrix(), self.basegrid[['N']].as_matrix(),lambda u, v: u-v)/self.bp
        a=vbpX
        b=vbpY
        self.vbp = np.dstack([a.ravel(),b.ravel()])[0].reshape(len(self.basepath),len(self.basegrid),2)

    
    def calculate_vbp(self):
        """ 
        :param self: vbp (vector bp)
        :return: None
        """

        try:
            a = np.stack([self.basepath[['E', 'N']].as_matrix()]*len(self.basegrid))
            b = np.stack([self.basegrid[['E', 'N']].as_matrix()]*len(self.basepath)).reshape(len(self.basegrid),len(self.basepath),2)

            self.a = a
            self.b = b
            # a little magic
            c = pd.DataFrame(np.vstack(a-b))
            c[0] = c[0]/ np.hstack(self.bp)
            c[1] = c[1]/ np.hstack(self.bp)

            self.vbp = c.as_matrix().reshape(len(self.basepath),len(self.basegrid),2)

        except RuntimeError:
            print('Runtime error is raised')
            raise

    def calculate_pbc(self):
        def foo(x):
                return cdist(np.matrix(self.vbc[x]),self.vbp[x,:],lambda u,v: np.arccos(np.dot(u,v)))[0]

        try:
            
            self.vbc = self.basepath.vbc.as_matrix()
            self.vbc[0] = [np.nan,np.nan]
            #vbc = self.vbc[1:]
            self.pbc = [foo(x) for x in range(len(self.vbc))]
            self.pbc = np.stack(self.pbc)
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def calculate_pto(self):
        try:
            vbc = self.basepath.vbc.as_matrix()
            self.mtp1 = vbc[1:] * np.cos(self.pbc[1:]).T
            self.mtp2 = np.multiply(self.bp[:-1].T,self.mtp1)
            self.coord = self.basepath[['E','N']].as_matrix()
            self.pto = map(lambda x: self.coord[:-1][x]+ list(self.mtp2.T[x]),range(len(self.coord[:-1])))
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def vmod(self,x):

        return np.sqrt((x*x).sum(axis=1))

    def set_index_correlation(self):

        try:
            coord = self.basepath[['E', 'N']].as_matrix()
            # Distancia Pto - B
            cd1 = map(lambda i: np.array(self.vmod(coord[:-1][i]- self.pto[i])), range(len(coord[:-1])))
            # Distancia Pto - C
            cd2 = map(lambda i: np.array(self.vmod(coord[1:][i] - self.pto[i])), range(len(coord[1:])))
            # Distancia C - Ba
            cd3 = self.vmod(coord[1:] - coord[:-1])
            # Refatorando o vetor cd3 para o tamanho de cd4
            cd3_2 = np.stack([cd3] * len(self.basegrid))
            # Soma de cd1 e cd2. (Deve ser igual a distancia cd3)
            cd4 = np.array(cd1) + np.array(cd2)
            # cd4 - cd3 (soma cd1 cd2)
            cd5 = pd.DataFrame(cd4 - cd3_2.T)
            # Distanca
            bp = pd.DataFrame(self.bp)

            self.cd1 = cd1
            self.cd2 = cd2
            self.cd3 = cd3
            self.cd4 = cd4
            self.cd5 = cd5

            self.basegrid.loc[:,'id_basepath'] = bp[bp < self.maximum_search_parameter][self.cd5<self.search_tolerance].idxmin()#cd5[pd.DataFrame(self.bp) < 400][cd5<0.00000001].idxmin()
            self.basegrid.loc[:,'id_basepath'][self.basegrid['id_basepath'].isnull()] = bp[bp < self.maximum_search_parameter].idxmin()
            self.basegrid.loc[:,'id_basepath'][self.basegrid['id_basepath'].isnull()] = 0
            self.basegrid.loc[:,'id_basepath'] = self.basegrid['id_basepath'].astype(int)

        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_pto(self):

        try:
            self.basegrid.loc[:,('ptoE')] = np.nan
            self.basegrid.loc[:,('ptoN')] = np.nan

            #pto = np.array(self.pto).reshape(len(self.basepath),len(self.basegrid))
            self.basegrid.loc[:,['ptoE', 'ptoN']] = np.array(self.pto)[self.basegrid['id_basepath']-1, range(len(self.basegrid['id_basepath']))]
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_position(self):
        try:
            bE = np.stack([self.basepath.E[:-1]] * len(self.basegrid)).T
            bN = np.stack([self.basepath.N[:-1]] * len(self.basegrid)).T
            cE = np.stack([self.basepath.E[1:]] * len(self.basegrid)).T
            cN = np.stack([self.basepath.N[1:]] * len(self.basegrid)).T
            pE = np.stack([self.basegrid.E] * len(cE))
            pN = np.stack([self.basegrid.N] * len(cE))
            pst = np.sign(np.multiply(cE - bE, pN - bN) - np.multiply(cN - bN, pE - bE))
            position = np.array(pst)[self.basegrid.id_basepath-1,range(len(self.basegrid))]
            self.basegrid.position = position
        except RuntimeError:
            print('Runtime error is raised')
            raise

    def set_d(self):
        try:
            idb = self.basegrid['id_basepath']
            d = np.array(self.bp)[idb,range(len(idb))] * np.sin(np.array(self.pbc)[idb, range(len(idb))])
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

        plt.figure(figsize=(10, 10))
        plt.scatter(self.basegrid.s[self.basegrid.s.notnull()], self.basegrid.d[self.basegrid.s.notnull()],
            c = self.basegrid.d[self.basegrid.s.notnull()],edgecolor='face')
        plt.show()

    def plot_result(self):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.basegrid.E, self.basegrid.N, c = self.basegrid.s, edgecolor='face')
        plt.scatter(self.basepath.E, self.basepath.N, c = self.basepath.Dist, edgecolor='face')
        plt.show()

    def plot_final(self):
        import matplotlib.gridspec as gridspec

        def plot_width(df,buff=100):
            return (df.d.astype('float')>-buff)&(df.d.astype('float')<buff)

        fig = plt.figure(figsize=(9.5,6))
        gs = gridspec.GridSpec(2,1,height_ratios=[4,1])

        ax0 = plt.subplot(gs[0])
        p0 = plt.scatter(
                        self.basegrid.E[plot_width(self.basegrid)].tolist(),
                        self.basegrid.N[plot_width(self.basegrid)].tolist(),
                        s=3,
                        edgecolors='none',
                        )
        p01 = plt.plot(
                        self.basepath.E,
                        self.basepath.N,
                        'black',
                        ls='-.',
                        lw=1,
                        label='s - line')
        plt.legend()
        ax0.set_xlabel('E (m)')
        ax0.set_ylabel('N (m)')

        ax1 = plt.subplot(gs[1])
        p0 = plt.scatter(
                        self.basegrid.s[plot_width(self.basegrid)].tolist(),
                        self.basegrid.d[plot_width(self.basegrid)].tolist(),
                        s=3,
                        edgecolors='none',
                        )
        plt.plot([0,self.basepath.Dist.max()],[0,0],'black',ls='-.',lw=1)
        ax1.set_xlabel('s (m)')
        ax1.set_ylabel('d (m)')
        plt.show()

    def run(self, progressbar = None):

        from datetime import datetime
        startTime = datetime.now()
        global progress_counter
        progress_counter = 0
        
        def counter(progress):

            if progress != None:
                global progress_counter
                progress_counter+=1
                progress.setValue(progress_counter)
            else:
                pass


        print datetime.now() - startTime
        counter(progressbar)
        self.calculate_bp()
        print(' calculate_bp')
        gc.collect()
        print datetime.now() - startTime
        counter(progressbar)
        self.calculate_vbp_old()
        print(' calculate_vbp')
        print datetime.now() - startTime
        gc.collect()
        self.calculate_pbc()
        counter(progressbar)
        print(' calculate_pbc')
        print datetime.now() - startTime
        gc.collect()
        self.calculate_pto()
        counter(progressbar)
        print(' calculate_pto')
        print datetime.now() - startTime
        gc.collect()
        self.set_index_correlation()
        counter(progressbar)
        print(' set_index_correlation')
        print datetime.now() - startTime
        gc.collect()
        self.set_pto()
        print(' set_pto')
        counter(progressbar)
        print datetime.now() - startTime
        gc.collect()
        self.set_position()
        counter(progressbar)
        print(' set_position')
        print datetime.now() - startTime
        gc.collect()
        self.set_d()
        counter(progressbar)
        print(' set_d')
        print datetime.now() - startTime
        gc.collect()
        self.set_s()
        counter(progressbar)
        print(' set_s')
        print datetime.now() - startTime
        gc.collect()
        self.plot_final()
        counter(progressbar)
        print datetime.now() - startTime
        gc.collect()








