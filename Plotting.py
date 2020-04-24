import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
import tensorflow as tf
import tensorflow_probability as tfp

from matplotlib import cm
from matplotlib.colors import LogNorm

from Configuration import centers

plt.rcParams['figure.dpi'] = 300.0
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 6

def plot_trans_ms(transM, init_vec, steps, annot=None):
  """

  suggested annot 's=rto $t_{' + str(t_step) + '}\\approx %s$ min' % str(np.round(t_segs[t_step-1]/60, 1)),

  """
  steps = np.atleast_1d(steps)
  maxstep = np.max(steps)
  transTe = tf.stack((tf.squeeze(transM.T),) * maxstep, axis=0)
  # transTe = tf.convert_to_tensor(((transM.T,) * maxstep))
  # transTe = tfp.math.dense_to_sparse(transTe)
  compoundTrans = tf.scan(lambda a, b: tf.matmul(a, b, a_is_sparse=True, b_is_sparse=True), transTe)
  l = len(steps) + 1
  l_init = len(init_vec)
  fig = plt.figure()
  axes = []
  for i in range(l):
    t_step = steps[i-1] + 1
    axes.append(fig.add_subplot(1, l, i + 1))
    # plt.subplot(1, l, i + 1)
    if i == 0:
      M = np.c_[init_vec[:, None], init_vec[:, None]]
      plt.text(
        x=l_init/2,
        s=r'to $t_{' + str(steps[i-1] + 1) + '}=$',
        y=np.round(l_init * 1.15),
        horizontalalignment='center'
      )
    else:
      if annot:
        plt.text(
          x=l_init/2,
          s=r'to $t_{' + str(t_step) + '}\\approx %s$ min' % str(np.round(t_segs[t_step-1]/60, 1)),
          y=np.round(l_init * 1.15),
          horizontalalignment='center'
        )
      # axes[i].set_title(r'transition to t_' + str(i), y=-0.01)
    M = np.array(np.squeeze(compoundTrans[t_step-2, ...]))
    # plt.imshow(M, aspect=('equal' if bool(i) else 0.4))
    M[M < 1e-5] = 1e-5
    plt.imshow(M, cmap=cm.get_cmap('plasma'), norm=LogNorm(vmin=1e-5, vmax=1))
  axes[0].set_xticks([])
  return fig, axes

def plot_sim_fractogram(model, states, **kwargs):
  for component, stateM in zip(model.components.index, states):
    if 'label' not in kwargs:
      plt.plot(centers(model.t_lims), np.diff(stateM[-1, :]), label=f'{component}', **kwargs)
    else:
      plt.plot(centers(model.t_lims), np.diff(stateM[-1, :]), **kwargs)
    plt.xlabel(r'time (s)')
    plt.ylabel(r'n (mol)')
