import numpy as np
from scipy.stats import entropy

def distr_lim_segmentation(distr, tr_lims):
  cp = distr.cdf(tr_lims)
  seg_p = np.diff(cp)
  return seg_p

def kl_divergence_vs_distr(pred_distr, pred_tr_lims, distr):
  kl_divergences = list()
  for dis, tr_lims in zip(pred_distr, pred_tr_lims):
    exp_distr_disc = distr_lim_segmentation(distr, tr_lims)
    kl_divergences.append(entropy(dis, exp_distr_disc))
  return kl_divergences

def analytical_var_noneq_isocratic(chi, w, t_r, R, D):
  return chi * w**2. * t_r / (R*D)

def analytical_rt(w, D, Vexp, V0, vdc, vdo):
  return w**2. / (6. * D) * np.log(1. + (Vexp * vdc/(V0 * vdo)))
