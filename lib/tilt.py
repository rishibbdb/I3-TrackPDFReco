import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jaxlib
import jax.numpy as jnp
from typing import List, Tuple, Any
import os
import jax
jax.config.update("jax_disable_jit", True)


tilt = 0
LMAX = 80
LYRS = 172
CTX = 11
CTY = 10
DIR1 = 9.3
DIR2 = 129.3
zoff = 1948.07
cv = np.pi/180
dir="/mnt/scratch/baburish/TPN-training/final/TPN_God/examples"
class tiltclass:
    
    def __init__(self):
        self.tilt = False
        self.lnum = 0
        self.l0 = 0
        self.r0 = 0
        self.tmod = 0
        self.LMAX = 80
        self.LYRS = 172
        self.CTX = 11
        self.CTY = 10
        self.DIR1 = 9.3
        self.DIR2 = 129.3
        self.zoff = 1948.07
        self.cv = np.pi/180

        self.mdir = []
        self.mmin = [] 
        self.mstp = []
        self.mnum = []
        self.mcut = []
        inFile=dir+"/tilt.set"
        if os.path.exists(inFile):
            # print("Tilt.set exists")
            with open(dir+"/tilt.set", "r") as file:
                n=0
                for line in file:
                    row = list(map(float, line.split()))
                    if len(row) == 5:
                        self.mdir.append(row[0])
                        self.mmin.append(row[1])
                        self.mstp.append(row[2])
                        self.mnum.append(row[3])
                        self.mcut.append(row[4])
                        n +=1
                    if n>=2:
                        break  
            if(self.mdir[0]!=self.DIR1 or self.mdir[1]!=self.DIR2):
                print("Unsupported tilt grid configuration")
            proj = np.ones((2, 2))
            proj[0][0] = np.cos(self.cv  * self.mdir[0])
            proj[0][1] = np.sin(self.cv  * self.mdir[0])
            proj[1][0] = np.cos(self.cv  * self.mdir[1])
            proj[1][1] = np.sin(self.cv  * self.mdir[1])
            det = proj[0][0] * proj[1][1] - proj[0][1] * proj[1][0]
            # print("Tilt Init: proj0, proj1, proj2, proj3",proj[0][0],proj[0][1],proj[1][0],proj[1][1] )
            # print("Tilt Init: det =", det)
            self.invproj = np.eye(2,2 )
            self.invproj[0][0] = proj[1][1]/det
            self.invproj[0][1] = -proj[0][1]/det
            self.invproj[1][0] = -proj[1][0]/det
            self.invproj[1][1] = proj[0][0]/det

            self.size = 0
            j = 0
            i = 0
            while(j<self.mnum[1]):
                while(i<self.mnum[0]):
                    if(i<j + self.mcut[0] and j < i+self.mcut[1]):
                        self.size += 1
                    i = i + 1
                j = j + 1

            if(self.size > self.LMAX):
                print("File tilt.set defines too many dust maps")
            if(self.mnum[0] > self.CTX or self.mnum[1] > self.CTY):
                print("Error: tilt map conifguration exceeds buffer")
            if(self.mstp[0] <= 0 or self.mstp[1] <= 0):
                print("Tilt map does not use increasing range order")

            m = 0
            j = 0
            i = 0
            self.mcol = np.eye(self.CTX, self.CTY)
            while(j < self.mnum[1]):
                while(i < self.mnum[0]):
                    if(i < j + self.mcut[0] and j < i + self.mcut[1]):
                        self.mcol[i][j] = m
                        m += 1
                    else:
                        self.mcol[i][j] = -1
                    i += 1
                j += 1 

            inFilemap=dir+"/tilt.map"
            if os.path.exists(inFilemap):
                # print("Tilt.map exists")
                with open(dir+"/tilt.map", "r") as file:
                    self.lnum = self.size
                    pts = np.zeros(self.lnum)
                    ds = []
                    vlp = [[] for _ in range(self.lnum)]
                    
                    for line in file:
                        row = line.split()
                        depth = float(row[0])
                        i = 0
                        for value in row[1:]:
                            pts[i] = float(value)
                            i += 1
                            if i >= self.lnum:
                                break
                        if i != self.lnum:
                            continue
                        ds.append(depth)
                        # print("Tilt Init: pts=", pts)
                        # print("Tilt Init: depth=", ds)
                        for i in range(self.lnum):
                            vlp[i].append(pts[i])
                        # print("Tilt Init: vlp=", vlp)

                self.size = len(ds)
                if(self.size-1>self.LYRS):
                    print("File tilt.map defines too many map points")
                
                i = 1
                while(i < self.size):
                    if(ds[i] < ds[i-1]):
                        print("Tilt map does not use increasing depth order")
                    i += 1 
                
                i = 0
                # self.lp = np.zeros(shape=(len(vlp),self.size))
                self.lp = np.zeros(shape=(LMAX, LYRS))
                while(i < self.lnum):
                    j = 0
                    while(j < self.size):
                        self.lp[i][j] = vlp[i][self.size-1-j]
                        j += 1
                    i += 1
                self.lpts=self.size-2

                if(self.size<2):
                    self.lnum=0
                else:
                    vmin = ds[0]
                    vmax = ds[self.size-1]
                    self.lmin = self.zoff - vmax
                    self.lrdz = (self.size - 1)/(vmax - vmin)
                    # print("DS", ds[self.size-1])
                    # print("DS size", self.size )
                    # print("lmin", self.lmin)
                    # print("lrdz", self.lrdz )
                # print("lnum=",self.lnum)

        if(self.lnum > 0):
            self.tmod=2
            # print('Loaded ' + str(self.lnum) + ' x ' + str(self.lpts) +' 2d dust layer points')
        else:
            thx = 225
            self.lnx = np.cos(self.cv *thx)
            self.lny = np.sin(self.cv *thx)
            # print("LNX and LNY", self.lnx, self.lny)
            ###Tilt par
            inFile_par=dir+'/tilt.par'
            if os.path.exists(inFile_par):
                # print("Tilt.par exists")
                with open(inFile_par, "r") as inFile:
                    #l0 = None
                    vlr = []
                    for line in inFile:
                        print(line.split())
                        try:
                            str_value, aux = map(float, line.split('/'))
                            if aux==0:
                                self.l0 = int(str_value)
                            vlr.append(aux)
                        except:
                            continue

                self.size = len(vlr)
                if self.size>self.LMAX:
                    print("File tilt.par defines too many dust maps")
                i=1
                while (i<self.size):
                    if vlr[i] < vlr[i-1]:
                        print("Tilt map does not use increasing range order")
                        i += 1
                i=0
                while(i<self.size):
                    self.lr[i] = vlr[i]
                
                inFile_data=dir+'/tilt.dat'
                if os.path.exists(inFile_data):
                    # print("Tilt.dat exists")
                    with open(inFile_data, 'r') as inFile:
                        self.lnum = self.size
                        pts = np.zeros(self.lnum)
                        ds = []
                        vlp = [[] for _ in range(self.lnum)]
                        for line in inFile:
                            dat_values = list(map(float, line.split()))
                            depth = dat_values[0]
                            pts = dat_values[1:self.lnum + 1]
                            # if len(dat_values < self.lnum+1):
                            #     break
                            # if len(pts) !=self.lnum:
                            #     break
                            ds.append(depth)
                            for i in range(self.lnum):
                                vlp[i].append(pts[i])

                    self.size = len(ds)
                    if(self.size-1>self.LYRS):
                        print("File tilt.dat defines too many map points")
                    
                    i = 1
                    while(i < self.size):
                        if(ds[i] < ds[i-1]):
                            print("Tilt map does not use increasing depth order")
                        i += 1
                    i = 0
                    # self.lp = np.zeros(shape=(len(vlp),self.size))
                    self.lp = np.zeros(shape=(LMAX, LYRS))
                    while(i < self.lnum):
                        j = 0
                        while(j < self.size):
                            self.lp[i][j] = vlp[i][self.size-1-j]
                            j += 1
                        i += 1
                    self.lpts=self.size-2
                    # print("DS size", self.size )
                    if(self.size<2):
                        self.lnum=0
                    else:
                        # print("DS", ds[self.size-1])
                        # print("DS size", self.size )
                        vmin = ds[0]
                        vmax = ds[self.size-1]
                        self.lmin = self.zoff - vmax
                        self.lrdz = (self.size - 1)/(vmax - vmin)
                        #print("lnum=",lnum)
            # print("Tilt Init: lnum=", self.lnum)
            if(self.lnum > 0):
                self.tmod=1
                # print('Loaded ' + str(self.lnum) + ' x ' + str(self.lpts) +' dust layer points')
            else:
                print("Not enough ice tilt points")

        tilt=True
        self.IceLayerTiltset_r0()

    def IceLayerTiltset_r0(self):
        if self.l0 == 86:
            self.r0=3.00521
            # print("r0 is set to ", self.r0)

    def IceLayerTiltJAX(self, x):#, self.tmod):
        rx = x[0]
        ry = x[1]
        rz = x[2]
        llp = jnp.array(self.lp)
        mn = jnp.array(self.mnum)
        mc =  jnp.array(self.mcol)
        result = 0
        z = (rz - self.lmin) * self.lrdz
        k = jnp.minimum(jnp.maximum(jnp.floor(z).astype(int), 0), self.lpts)
        l = k + 1
        fraction_z_above = z - k
        fraction_z_below = l - z

        midp = lambda z_correction_above, z_correction_below: (z_correction_above * fraction_z_above + z_correction_below * fraction_z_below)

        if(self.tmod==1):
            # print("Tilt Layer: lnum=",self.lnum)
            # print("Tilt Layer: lnx, lny=", self.lnx, self.lny)
            nr = self.lnx * rx + self.lny * ry - self.r0
            j=1
            while j < self.LMAX:
                if nr < self.lr[j] or j == self.lnum - 1:
                    break
                j += 1
            i = j - 1
            result = ( midp(llp[j][l],llp[j][k]) * (nr-self.lr[i]) + midp(llp[i][l],llp[i][k]) * (self.lr[j]-nr) )/(self.lr[j]-self.lr[i] ) 

        elif(self.tmod==2):
            # print("Tilt Layer (tmod=2): lnum=",self.lnum)
            qx = self.invproj[0][0] * rx + self.invproj[1][0] * ry
            qy = self.invproj[0][1] * rx + self.invproj[1][1] * ry
            qx=(qx - self.mmin[0])/self.mstp[0]
            qy=(qy - self.mmin[1])/self.mstp[1]
            xx = qx
            yy = qy
            n0 = self.mnum[0] - 1
            n1 = self.mnum[1] - 1
            c0 = self.mcut[0] - 1
            c1 = self.mcut[1] - 1

            cond1 = yy < 0
            cond2 = jnp.logical_and( xx - yy > 0, xx <= c0)
            c_cond1 = jnp.logical_and(cond1, cond2)
            xx_new = jnp.dot((xx - yy), c0 / (c0 - yy))
            yy_new = 0
            xx = jnp.where(c_cond1, xx_new, xx)
            yy = jnp.where(c_cond1, yy_new, yy)

            cond1 = xx < 0 
            cond2 = jnp.logical_and(yy - xx >= 0, yy < c1)
            c_cond2 = jnp.logical_and(cond1, cond2)
            xx_new = 0
            yy_new = jnp.dot((yy - xx) ,c1/(c1 - xx))
            xx = jnp.where(c_cond2, xx_new, xx)
            yy = jnp.where(c_cond2, yy_new, yy)

            cond1 = yy>n1
            cond2 = jnp.logical_and(xx >= n1 - c1, yy - xx > n1 - n0)
            c_cond3 = jnp.logical_and(cond1, cond2)
            xx_new = n1 - c1 + jnp.dot((n0  - (n1 - c1)), (xx - (n1 - c1))/(yy - (n1 - n0) - (n1 - c1)))
            yy_new = n1
            xx = jnp.where(c_cond3, xx_new, xx)
            yy = jnp.where(c_cond3, yy_new, yy)

            cond1 = xx>n0
            cond2 = jnp.logical_and(yy > n0-c0, yy-xx<=n1-n0)
            c_cond4 = jnp.logical_and(cond1, cond2)
            xx_new = n0
            yy_new = n0 - c0 + jnp.dot((n1 - (n0 - c0)), (yy - (n0 - c0))/(xx - (n0 - n1) - (n0 - c0)))
            xx = jnp.where(c_cond4, xx_new, xx)
            yy = jnp.where(c_cond4, yy_new, yy)

            cond1 = yy - xx > c1
            cond2 = jnp.logical_and(yy >= c1 , xx < n1 - c1)
            c_cond5 = jnp.logical_and(cond1, cond2)
            xy = yy - xx - c1
            xy = c1 + jnp.dot((xx + yy - c1 + xy ), (n1 - c1)/(n1 - c1 + xy ))
            xx_new = (xy - c1)/2
            yy_new = (xy + c1)/2
            xx = jnp.where(c_cond5, xx_new, xx)
            yy = jnp.where(c_cond5, yy_new, yy)

            cond1 = xx - yy > c0
            cond2 = jnp.logical_and(xx >= c0 , yy <=n0 - c0)
            c_cond6 = jnp.logical_and(cond1, cond2)
            xy = xx - yy - c0
            xy = c0 + jnp.dot((xx + yy - c0 + xy ), (n0 - c0)/(n0 - c0 + xy ))
            xx_new = (xy + c0)/2
            yy_new = (xy - 0)/2
            xx = jnp.where(c_cond6, xx_new, xx)
            yy = jnp.where(c_cond6, yy_new, yy)

            qx = xx 
            qy = yy

            cell_nx = jnp.clip(jnp.floor(qx).astype(int), 0, self.mnum[0]).astype(int)
            cell_ny = jnp.clip(jnp.floor(qy).astype(int), 0, self.mnum[1]).astype(int)

            relative_cellx = qx - cell_nx
            relative_celly = qy - cell_ny

            qrx = jnp.where(relative_cellx>relative_celly, relative_cellx, relative_celly)
            qry = jnp.where(relative_cellx>relative_celly, relative_celly, relative_cellx)
            qnx = jnp.where(relative_cellx>relative_celly,cell_nx + 1, cell_nx)
            qny = jnp.where(relative_cellx>relative_celly, cell_ny, cell_ny + 1)

            def findcol(x, y):
                cond1 = jnp.logical_and(x>=0, x<mn[0])
                cond2 = jnp.logical_and(y>=0, y<mn[1]) #(x >= 0) & (x < mnum[0]) & (y >= 0) & (y < mnum[1])
                comb_cond = jnp.logical_and(cond1, cond2)
                col_res = jnp.where(comb_cond, mc[x, y], -1)
                return col_res

            cellx = findcol(cell_nx, cell_ny)
            celly = findcol(qnx, qny)
            cellz = findcol(cell_nx + 1, cell_ny + 1)
            weightx = 1 - qrx
            weighty = qrx - qry
            weightz = qry
            
            result1 = jnp.where(cellx < 0, 0, midp(llp[cellx.astype(int)][l], llp[cellx.astype(int)][k]) * weightx)
            result2 = jnp.where(celly < 0, 0, midp(llp[celly.astype(int)][l], llp[celly.astype(int)][k]) * weighty)
            result3 = jnp.where(cellz < 0, 0, midp(llp[cellz.astype(int)][l], llp[cellz.astype(int)][k]) * weightz)

            result = result1 + result2 + result3
        return result