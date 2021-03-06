from pylab import *;
import diffusion;
import forcing;
import pSpectral;
import RungeKutta;

ion()

Nx=256;
Ny=128;
xx = linspace(-pi,pi-2*pi/Nx,Nx);
yy = linspace(-pi,pi-2*pi/Ny,Ny);
[y,x]=  meshgrid(xx,yy);
a = sin(30*x)+sin(40*x)*cos(50*y);



diff = diffusion.specDiffusion(Nx,Ny, alpha=0, nu=1e-6);
p = pSpectral.parSpectral(Nx,Ny);

def dfdt(t,f,arge=None):
    return 0.1*p.laplacian(f);
    #return f-f;
dt=0.001;

def dfsn(dt,f):
    return f;


delta = 2*pi/max(Nx,Ny);

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, dfsn ,1);

t = 0;
f = a;



while (t<50):
	tnew,fnew = stepfwd.integrate(t,f,dt);
	t = tnew;
	f = fnew;
	imshow(fnew);
	colorbar();
	pause(1e-3);
	clf();
