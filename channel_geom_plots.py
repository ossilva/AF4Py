import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Configuration import Configuration

plt.rcParams['font.family'] = 'serif'
c = Configuration('Wyatt SC')
z_range = np.linspace(c.z_origin, c.L, 200)
b = np.vectorize(lambda z: c.bz(z=z))

def plate_surf(first_c, second_c, const, dim, ax, **kwargs):
# face,left,right,top
# 1,2,5,3
  len_c = len(first_c)
  fir_m = np.outer(first_c, np.ones(len_c))
  sec_m = np.outer(second_c, np.ones(len_c)).T
  const_m = np.outer(np.repeat(const, len_c), np.ones(len_c))
  ms = [fir_m, sec_m]
  ms.insert(dim, const_m)
  return ax.plot_surface(*ms, **kwargs)

def surf_box(z_b, wi, he, le, ax, tops=True, **kwargs):
  plate_surf([0, le], [z_b, z_b+he], -wi/2, 1, ax, **kwargs)
  plate_surf([-wi/2, wi/2], [z_b, z_b+he], le, 0, ax, **kwargs)
  # plate_surf([-wi/2, wi/2], [0, he], 0, 0, ax, **kwargs)
  if tops:
    plate_surf([0, le], [-wi/2, wi/2], z_b, 2, ax, **kwargs)
    plate_surf([0, le], [-wi/2, wi/2], z_b+he, 2, ax, **kwargs)

br = 50
he = 5
sp = 11
pt_he = 4
bpt_he = pt_he/2
#
my_mpl_kws = {'color': 'g', 'alpha': 0.32, 'linewidth': 0}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
### spacer cover
# y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
# x = np.c_[[z_range]*2].T
# z = np.ones(len(z_range))[:, None] * np.array([0, 0])[None, :]
# ax.plot_surface(x, y, z, **my_mpl_kws)
# #
# y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
# x = np.c_[[z_range]*2].T
# z = np.ones(len(z_range))[:, None] * np.array([0, 0])[None, :]
# ax.plot_surface(x, -y, z, **my_mpl_kws)
# #
# y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
# x = np.c_[[z_range]*2].T
# z = np.ones(len(z_range))[:, None] * np.array([he, he])[None, :]
# ax.plot_surface(x, y, z, **my_mpl_kws)
# #
# y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
# x = np.c_[[z_range]*2].T
# z = np.ones(len(z_range))[:, None] * np.array([he, he])[None, :]
# ax.plot_surface(x, -y, z, **my_mpl_kws)
### spacer cover end
### inside line
zo = c.L - c.zL
x = np.array([0, c.z0, zo, c.L] + [0, c.z0, zo][::-1])
x = np.repeat(x, 2)
x = np.tile(x, 2)
y = np.concatenate([np.array([0, b(c.z0), b(zo), 0]), -np.array([0, b(c.z0), b(zo)][::-1])])
y = np.repeat(y, 2)
y = np.tile(y, 2)
z = np.repeat([0, he], 2)
z = np.tile(z, 7)
z = np.roll(z, 1)
my_mpl_kws = {'color': 'b', 'alpha': 0.32}
ax.plot(x, y, z, **my_mpl_kws)
#
plate_surf([0, c.L], [0, he], br/2, 1, ax, **my_mpl_kws)
plate_surf([0, c.L], [0, he], -br/2, 1, ax, **my_mpl_kws)
plate_surf([-br/2, br/2], [0, he], c.L, 0, ax, **my_mpl_kws)
plate_surf([-br/2, br/2], [0, he], 0, 0, ax, **my_mpl_kws)
### inside end
### inside
# zo = c.L - c.zL
# x = np.array([
#   [0, 0],
#   [c.z0, c.z0],
#   [zo, zo],
#   [c.L, c.L]
# ])
# y = np.outer(np.array([0, b(c.z0), b(zo), 0]), np.ones(2))
# z = np.outer(np.ones(4), np.array([he,0]))
# my_mpl_kws = {'color': 'b', 'alpha': 0.32}
# ax.plot_surface(x, y, z, **my_mpl_kws)
# ax.plot_surface(x, -y, z, **my_mpl_kws)
# plate_surf([0, c.L], [0, he], br/2, 1, ax, **my_mpl_kws)
# plate_surf([0, c.L], [0, he], -br/2, 1, ax, **my_mpl_kws)
# plate_surf([-br/2, br/2], [0, he], c.L, 0, ax, **my_mpl_kws)
# plate_surf([-br/2, br/2], [0, he], 0, 0, ax, **my_mpl_kws)
### inside end
### bot plate frit
frit_kws = {'color': 'g', 'alpha': 0.15, 'linewidth': 0}
y = b(z_range)[:, None] * np.array([0, 1])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp, -sp])[None, :]
ax.plot_surface(x, y, z, **frit_kws)
#
y = b(z_range)[:, None] * np.array([0, 1])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp, -sp])[None, :]
ax.plot_surface(x, -y, z, **frit_kws)
#
y = b(z_range)[:, None] * np.array([0, 1])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp-bpt_he, -sp-bpt_he])[None, :]
ax.plot_surface(x, y, z, **frit_kws)
#
y = b(z_range)[:, None] * np.array([0, 1])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp-bpt_he, -sp-bpt_he])[None, :]
ax.plot_surface(x, -y, z, **frit_kws)
### bot plate frit end
### bot plate cover
y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp, -sp])[None, :]
ax.plot_surface(x, y, z, **my_mpl_kws)
#
y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp, -sp])[None, :]
ax.plot_surface(x, -y, z, **my_mpl_kws)
#
y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp-bpt_he, -sp-bpt_he])[None, :]
ax.plot_surface(x, y, z, **my_mpl_kws)
#
y = b(z_range)[:, None] * np.array([0, 1])[None, :] + np.array([br/2, 0])[None, :]
x = np.c_[[z_range]*2].T
z = np.ones(len(z_range))[:, None] * np.array([-sp-bpt_he, -sp-bpt_he])[None, :]
ax.plot_surface(x, -y, z, **my_mpl_kws)
### bot plate cover end
### plates
# pt_he = 5
quiv_kws = {"color": "r", "arrow_length_ratio": 0.35,
            # "length": pt_he,
            "linewidths": 2.1
            }
surf_box(he+sp, br, pt_he, c.L, ax, **my_mpl_kws)
surf_box(-bpt_he-sp, br, bpt_he, c.L, ax, tops=False, **my_mpl_kws)
ax.quiver(c.z0, 0, he + sp + pt_he, 0, 0, -pt_he, pivot='tail', **quiv_kws)
ax.quiver(c.L, 0, he + sp + pt_he, 0, 0, pt_he, pivot='tip',  **quiv_kws)
ax.quiver(0, 0, he + sp + pt_he, 0, 0, -pt_he, pivot='tail',  **quiv_kws)
ax.quiver(c.L/2, 0, -sp, 0, 0, -pt_he, pivot='tail',  **quiv_kws)
ax.text(c.z0, 0, he + sp + pt_he, "(injection inlet)", va="top")
ax.text(c.L, 0, he + sp + pt_he, "detector outlet", ha="right")
ax.text(0, 0, he + sp + pt_he, "inlet")
ax.text(c.L/2, 0, -he - sp, "crossflow outlet")
#
X = np.array([0, c.L])
Z = np.array([-bpt_he-sp, he+sp+bpt_he])
Y = np.array([-br/2, br/2])
#
ax.set_zlim(-30, 30)
ratio = 0.5
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = ratio*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + ratio*(X.max()+X.min())
Yb = ratio*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + ratio*(Y.max()+Y.min())
Zb = ratio*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + ratio*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')
ax_sep = -20
quiv_kws = {"color": "black", "arrow_length_ratio": 0.21,
            # "length": pt_he,
            #"width": 4.
            }
ax.quiver(ax_sep, -br/2 + ax_sep, (-sp - bpt_he + ax_sep), 0, 0, 38, pivot='tail',  color='black', length=0.45, arrow_length_ratio=0.3)
ax.quiver(ax_sep, -br/2 + ax_sep, (-sp - bpt_he + ax_sep), 0, 38, 0, pivot='tail',  **quiv_kws)
ax.quiver(ax_sep, -br/2 + ax_sep, (-sp - bpt_he + ax_sep), 38, 0, 0, pivot='tail', **quiv_kws)
ax.text(ax_sep - ax_sep, -br/2 + ax_sep - ax_sep, (-sp - pt_he + ax_sep),  r"$z$", ha="left")
ax.text(ax_sep-5, -br/2 + ax_sep - 2*ax_sep, (-sp - pt_he + ax_sep),  r"$x$", ha="right")
ax.text(ax_sep, -br/2 + ax_sep, (-sp - pt_he + ax_sep) - ax_sep, r"$y$", ha="right")
# axis vector
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.65, 1.65, 1.65, 1]))
ax.set_axis_off()
ax.view_init(45, -75)
ax.set_xlim3d(0, 350)
fig.savefig('./gen_figs/chan_assemb.png', dpi=300,
            figsize=np.array([6.4, 4.8])*2,
            # transparent=True
            )
# plt.show()
# plt.close()

