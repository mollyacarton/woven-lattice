# Coded by Molly Carton MIT Sept 2022-Aug 2024
# Portela Research Group: Woven metamaterial lattice generator
# Ver. Aug. 2024


import makeunitcell_func1 as muc
import numpy as np

latName = 'diamond' #Lattice type: cubic, bcc, octahedron, tetrakai, diamond, (experimental: cuBCC,tetracubic,arrow,octet)
unitCell = 60 #Length of unit cell side
sampling = 2 #Approximate distance between sample points
cellNum = [3,2,1] #Number of unit cells in lattice
doubleNetworkFlag=True #Double network (concentric)
trimFlag=False #Trim to UC boundary

def strutParams(strutPosition,L):
    x,y,z=float(strutPosition[0]+L),float(strutPosition[1]+L),float(strutPosition[2]+L) #x,y,z coordinate
    xuc,yuc,zuc = int(round((x+L)/(2*L))), int(round((y+L)/(2*L))), int(round((z+L)/(2*L))) #x,y,z unit cell index
    strutRadius = 2
    strutRevolutions = 3/4
    return (float(strutRadius),float(strutRevolutions))



curves = muc.wovenLattice(latName,unitCell/2,strutParams,sampling,cellNum,doubleNetwork=doubleNetworkFlag,trim=trimFlag,defectPoints=[])

muc.plotCurves(curves,cellNum)
muc.exportCSV(curves,cellNum,unitCell/2,doubleNetwork=doubleNetworkFlag,trim=trimFlag)
muc.exportOneCSV(curves,cellNum,unitCell/2)
muc.save_piped_stl(curves,1,10,filename='woven_mesh.stl',eccentricity=1) #fiber radius, number of points in circular cross-section



quit()

