import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

Lx = 6.28; nx = 100
Ly = 6.28; ny = 100

x = np.linspace(0,Lx,nx)
y = np.linspace(0,Ly,ny)
X, Y = np.meshgrid(x,y)

h = (np.cos(X) * np.sin(Y)).reshape(nx*ny)
u = np.zeros((ny,nx-1)).reshape((nx-1)*ny)
v = np.zeros((ny-1,nx)).reshape(nx*(ny-1))

spatial_order = 6
assert spatial_order in [2,3,4,5,6]

print('Building ddx...', end=' ', flush=True)
start = time.time_ns()

data = []

for i in range(ny*(nx-1)):
    # for j in range(ny*nx):
    #     if j//nx == i//(nx-1):
    for j in range(nx*(i//(nx-1)), min(nx*(i//(nx-1))+nx, nx*ny)):
            if i%(nx-1) == j%nx or i%(nx-1) == j%nx-1:
                match spatial_order:
                    case 2 | 3:
                        data.append((i, j, 1/2))
                    case 4 | 5:
                        data.append((i, j, 7/12))
                    case 6:
                        data.append((i, j, 37/60))
            elif i%(nx-1) == j%nx+1 or i%(nx-1) == j%nx-2:
                match spatial_order:
                    case 4 | 5:
                        data.append((i, j, -1/12))
                    case 6:
                        data.append((i, j, -2/15))
            elif i%(nx-1) == j%nx+2 or i%(nx-1) == j%nx-3:
                match spatial_order:
                    case 6:
                        data.append((i, j, 1/60))

# ddx : (ny,nx) -> (ny,nx-1)
ddx = scipy.sparse.bsr_matrix(
    ([x[2] for x in data], ([x[0] for x in data], [x[1] for x in data])),
    shape=(ny*(nx-1), nx*ny))

duration = time.time_ns() - start
print(f'done ({duration/1e6:.4f} ms)')

print('Building ddy...', end=' ', flush=True)
start = time.time_ns()

data = []
for i in range((ny-1)*nx):
    # for j in range(ny*nx):
    #     if j%nx == i%nx:
    for j in range(i%nx, ny*nx+i%nx, nx):
        if i//nx == j//nx or i//nx == j//nx-1:
            match spatial_order:
                case 2 | 3:
                    data.append((i, j, 1/2))
                case 4 | 5:
                    data.append((i, j, 7/12))
                case 6:
                    data.append((i, j, 37/60))
        elif i//nx == j//nx+1 or i//nx == j//nx-2:
            match spatial_order:
                case 4 | 5:
                    data.append((i, j, -1/12))
                case 6:
                    data.append((i, j, -2/15))
        elif i//nx == j//nx+2 or i//nx == j//nx-3:
            match spatial_order:
                case 6:
                    data.append((i, j, 1/60))

# ddy : (ny,nx) -> (ny-1,nx)
ddy = scipy.sparse.bsr_matrix(
    ([x[2] for x in data], ([x[0] for x in data], [x[1] for x in data])),
    shape=((ny-1)*nx, nx*ny))

duration = time.time_ns() - start
print(f'done ({duration/1e6:.4f} ms)')

print('Calculating derivatives...', end=' ', flush=True)
start = time.time_ns()
dhdx = ddx @ h
dhdy = ddy @ h
duration = time.time_ns() - start
print(f'done ({duration/1e6:.4f} ms)')

plt.pcolormesh(x, y, h.reshape((ny,nx)), cmap='coolwarm')
x_half = (x[1:] + x[:-1]) / 2
plt.quiver(x_half, y,
           dhdx.reshape((ny,nx-1)), np.zeros_like(dhdx).reshape((ny,nx-1)),
           pivot='mid')
y_half = (y[1:] + y[:-1]) / 2
plt.quiver(x, y_half,
           np.zeros_like(dhdy).reshape((ny-1,nx)), dhdy.reshape((ny-1,nx)),
           pivot='mid')
plt.gca().set_aspect('equal')
plt.show()