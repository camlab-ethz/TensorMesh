from FEGNNToy.mesh.lagrange2dmesh import mesh2d
from FEGNNToy.fe.shapefun import shape2d
from FEGNNToy.fe.gaussrule import gausspoint2d
from FEGNNToy.postprocess.PlotResult import Plot2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def linearElasticity(E = 1.0e9, nu=0.3):
    mymesh=mesh2d(nx=40,ny=40,xmax=1.0,ymax=1.0,meshtype='quad4')
    mymesh.createmesh()
    mymesh.plotmesh()

    gpoints = gausspoint2d(ngp=4)
    gpoints.creategausspoint()

    shp = shape2d(meshtype='quad4')
    shp.update()

    nDofs = mymesh.nodes*2
    K = np.zeros((nDofs,nDofs))
    F = np.zeros(nDofs)

    B = np.zeros((3,2))
    Bt = np.zeros((2,3))
    D = np.zeros((3,3))

    D[0,0]=E/(1-nu**2)   ;D[0,1]=E*nu/(1-nu**2)
    D[1,0]=E*nu/(1-nu**2);D[1,1]=E/(1-nu**2)
    D[2,2]=0.5*E*(1-nu)/(1-nu**2)

    for e in range(mymesh.elements):
        elconn=mymesh.elementconn[e,:]
        nodes=mymesh.nodecoords[elconn,:]
        for gp in range(gpoints.ngp2):
            xi =gpoints.gpcoords[gp,1]
            eta=gpoints.gpcoords[gp,2]
            w  =gpoints.gpcoords[gp,0]
        
            shp_val,shp_grad,j=shp.calc(xi,eta,nodes[:,0],nodes[:,1])
            JxW=j*w
            for i in range(mymesh.nodesperelement):
                B[0,0]=shp_grad[i,0]
                B[1,1]=shp_grad[i,1]
                B[2,0]=shp_grad[i,1]
                B[2,1]=shp_grad[i,0]
                iInd=elconn[i]
                for j in range(mymesh.nodesperelement):
                    Bt[0,0]=shp_grad[j,0];Bt[0,2]=shp_grad[j,1]
                    Bt[1,1]=shp_grad[j,1];Bt[1,2]=shp_grad[j,0]
                    C=np.dot(Bt,np.dot(D,B)) # C=Bt*D*B==> 2x2 matrix
                    jInd=elconn[j]
                    # K_ux,ux
                    K[2*iInd+0,2*jInd+0]+=C[0,0]*JxW
                    # K_ux,uy
                    K[2*iInd+0,2*jInd+1]+=C[0,1]*JxW
                    # K_uy,ux
                    K[2*iInd+1,2*jInd+0]+=C[1,0]*JxW
                    # K_uy,uy
                    K[2*iInd+1,2*jInd+1]+=C[1,1]*JxW
    #########################################
    Penalty=1.0e16

    # fix ux=0 for bottom edge
    iInd=mymesh.bcnodeids['bottom']*2
    K[iInd,iInd]+=Penalty
    F[iInd]=0.0

    # fix uy=0 for bottom edge
    iInd=mymesh.bcnodeids['bottom']*2+1
    K[iInd,iInd]+=Penalty
    F[iInd]=0.0

    # apply uy=0.05 for top edge
    iInd=mymesh.bcnodeids['top']*2+1
    K[iInd,iInd]+=Penalty
    F[iInd]=0.05*Penalty

    disp=np.linalg.solve(K,F)

    NodeCount = np.zeros(mymesh.nodes)
    Stress_yy = np.zeros(mymesh.nodes)
    Strain_yy = np.zeros(mymesh.nodes)

    elu = np.zeros((mymesh.nodesperelement,2))

    for e in range(mymesh.elements):
        elconn=mymesh.elementconn[e,:]
        x = mymesh.nodecoords[elconn,0]
        y = mymesh.nodecoords[elconn,1]
        for i in range(mymesh.nodesperelement):
            iInd=elconn[i]
            elu[i,0]=disp[2*iInd]
            elu[i,1]=disp[2*iInd+1]
        gpstress_yy=0.0
        gpstrain_yy=0.0
        for gp in range(gpoints.ngp2):
            xi = gpoints.gpcoords[gp,1]
            eta= gpoints.gpcoords[gp,2]
            w  = gpoints.gpcoords[gp,0]
            shp_val,shp_grad,j=shp.calc(xi,eta,x,y)

            stress_yy=0.0
            strain_yy=0.0
            for i in range(mymesh.nodesperelement):
                B[0,0]=shp_grad[i,0]
                B[1,1]=shp_grad[i,1]
                B[2,0]=shp_grad[i,1]
                B[2,1]=shp_grad[i,0]
                strain=np.dot(B,elu[i,:])
                stress=np.dot(D,strain)
                stress_yy+=stress[1]
                strain_yy+=strain[1]

            gpstress_yy+=stress_yy/gpoints.ngp2
            gpstrain_yy+=strain_yy/gpoints.ngp2
        for i in range(mymesh.nodesperelement):
            iInd=elconn[i] # global index
            Stress_yy[iInd]+=gpstress_yy
            Strain_yy[iInd]+=gpstrain_yy
            NodeCount[iInd]+=1.0

    for i in range(mymesh.nodes):
        Stress_yy[i]/=NodeCount[i]
        Strain_yy[i]/=NodeCount[i]


    
    X, Y = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
    Ux = griddata((mymesh.nodecoords[:,0],mymesh.nodecoords[:,1]),disp[0::2],(X,Y),method='linear')
    Uy = griddata((mymesh.nodecoords[:,0],mymesh.nodecoords[:,1]),disp[1::2],(X,Y),method='linear')

    n_axis = int(np.sqrt(mymesh.nodes))
    dispX = disp[0::2].reshape(n_axis,n_axis)
    dispY = disp[1::2].reshape(n_axis,n_axis)
    
    x = mymesh.nodecoords[:,0].reshape(n_axis,n_axis)
    y = mymesh.nodecoords[:,1].reshape(n_axis,n_axis)

    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    img1 = ax[0].pcolormesh(x, y, dispX,  cmap="jet", shading="gouraud")
    img2 = ax[1].pcolormesh(x, y, dispY,  cmap="jet", shading="gouraud")
    img3 = ax[2].pcolormesh(x, y, Stress_yy.reshape(n_axis,n_axis),  cmap="jet", shading="gouraud")
    img4 = ax[3].pcolormesh(x, y, Strain_yy.reshape(n_axis,n_axis),  cmap="jet", shading="gouraud")
    cb1  = fig.colorbar(img1)
    cb2  = fig.colorbar(img2)
    cb3  = fig.colorbar(img3)
    cb4  = fig.colorbar(img4)
    ax[0].set_title("X Displacement")
    ax[1].set_title("Y Displacement")
    ax[2].set_title("Stress_yy")
    ax[3].set_title("Strain_yy")
    plt.show()

if __name__ == '__main__':
    linearElasticity()