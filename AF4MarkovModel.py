import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm

import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

from Configuration import centers

K_B = 1.38064852e-23 #si
AQ_ETA_20 = 8.90e-4 #si
T_20 = 293.15

def stokes_einstein(D, eta, temp):
 return K_B * temp / (6. * np.pi * eta * D)

class AF4MarkovModel():
  COMPONENT_INDEX = 'name'
  COMPONENT_VARS = ('n', 'D', 'alpha')

  @staticmethod
  def component_ell_from_vdc(D, vdcs, Atot):
    return D / (vdcs * Atot)

  def __init__(self, config):
    self.config = config
    self.components = pd.DataFrame(columns=self.COMPONENT_VARS)
    self.init_states = dict()

  def add_component(self, name, n, D, R=None, alpha=None):
    if not alpha:
      # Stokes-Einstein relation, alpha = r / w
      # uses parameters for water at 20C
      R = R or stokes_einstein(D, AQ_ETA_20, T_20)
      alpha = R / self.config.w
    entry = pd.Series((
        n,
        D,
        alpha,
    ), name=name, index=self.COMPONENT_VARS)
    self.components = self.components.append(entry)

  def set_init_state(self, entries):
    for key in entries.keys():
      if key not in self.components.index:
        raise Exception(f'{key} not in components')
    self.init_states.update(entries)

  def set_init_focus(self, percent):
    ns = self.components.n.to_numpy()
    z_focus = self.z_centers[
      np.digitize(percent / 100 * (self.config.L -  self.config.zL), self.z_lims) - 1]
    z_lims = self.z_lims
    mu = z_focus
    sigma = self.config.eq_zone_width_general(self.config.w)
    z_states = np.arange(len(z_lims), dtype=np.float64)
    transform = tfb.Chain([
     tfb.Shift(tf.constant(-1., tf.float64)),
     tfb.Scale(tf.constant(1., dtype=tf.float64)/np.diff(z_lims[:2]).item()),])
    truncNormal = tfp.distributions.TruncatedNormal(
        loc=mu,
        scale=sigma,
        low=self.config.z_origin,
        high=np.inf,
        name='p(z | t=t_0)')
    truncNormalsQuantized = tfd.QuantizedDistribution(
        distribution=tfd.TransformedDistribution(distribution=truncNormal,
                                                  bijector=transform),
        low=z_states[0],
        high=z_states[-1])  # last z_bin is P[Y>k-1]
    probs = truncNormalsQuantized.prob(z_states)
    entries = dict(zip(
          self.components.index.to_list(),
          list(probs[None, :] * ns[:, None])))
    self.set_init_state(entries)

  @staticmethod
  def compute_transition_matrix(propDF, origin, z_states, z_increment):
    num_states = len(z_states)
    seg_cp = np.empty((num_states, num_states, len(propDF)))
    for i, (_, row) in enumerate(propDF.iterrows()):
      print(f'calculating transition matrix {i+1}/{len(propDF.index)}')
      propM = row.propM
      if np.isnan(propM).sum() > 1:
        raise Exception("improper limits for model")
      mu = (propM[:, 0] + propM[:, 1])
      sigma = np.sqrt(propM[:, 2])
      # quantizing uses ceiling()
      transform = tfb.Chain([
       tfb.Shift(tf.constant(-1., tf.float64)), tfb.Scale(1./z_increment)])
      truncNormals = tfp.distributions.TruncatedNormal(
          loc=mu,
          scale=sigma,
          low=origin,
          high=np.inf,
          name='p(z | z_-1=z\', t)')
      truncNormalsQuantized = tfd.QuantizedDistribution(
          distribution=tfd.TransformedDistribution(distribution=truncNormals,
                                                   bijector=transform),
          low=z_states[0],
          high=z_states[-1])  # last z_bin is P[Y>k-1]
      probs = truncNormalsQuantized.prob(z_states[:, None])
      seg_cp[:, :-1, i] = probs
      seg_cp[:, -1, :] = 0.
      seg_cp[-1, -1, :] = 1.
    return seg_cp

  def compute_model_params(self, z_lims, use_rk=True):
    self.z_lims = z_lims
    self.z_centers = centers(z_lims)
    self.t_lims = self.config.time_lims
    z_increment = np.diff(z_lims[:2])
    if not np.all(np.isclose(np.diff(z_lims), z_increment, 1e-10)):
      raise Exception('only equipartitioned length currently supported')
    z_states = np.round(z_lims / z_increment, 0)
    transitionDi = dict()
    ellDi = dict()
    for i, (idx, component) in enumerate(self.components.iterrows()):
      print(f'computing params for component {i+1}/{len(component)}')
      [_, D, alpha] = component.to_list()
      t_id_dict, propDF = self.config.zt_linear_flow_props(z_lims, alpha, D, use_rk)
      if i == 0:
        self.t_id_dict = t_id_dict
      vdcs = propDF['V_c'].loc[t_id_dict.values()]
      ellDi[idx] = self.component_ell_from_vdc(D, vdcs, self.config.Atot)
      transitionDi[idx] = self.compute_transition_matrix(
          propDF, self.config.z_origin, z_states, z_increment)
    self.transTensorDi = transitionDi
    self.ellDi = ellDi
    self.seg_vols = self.config.segment_Vs(z_lims)

  def state_distributions_to_t_idx(self, names=None, t_idx=None):
    if t_idx is None:
      t_idx = len(self.t_lims)-1
    names = np.atleast_1d(names) or self.components.index
    # include init_state
    state_p_t = np.empty((len(names), len(self.z_lims) , t_idx + 1))
    for i, name in enumerate(names):
      init_state = self.init_states[name]
      init_state = init_state / np.sum(init_state)
      transTe = self.transTensorDi[name]
      idx_seq = list(self.t_id_dict.values())
      transTe = transTe[..., idx_seq[:t_idx]]
      transTe = tf.transpose(transTe, perm=[2, 0, 1])
      # transTe = tfp.math.dense_to_sparse(transTe)
      # len(t_idx) x len(z_lims)**2
      compoundTrans = tf.scan(lambda a, b: tf.matmul(a, b, a_is_sparse=True, b_is_sparse=True), transTe)
      # batch dim broadcasted
      state_t = init_state[None, None, :] @\
        tf.transpose(compoundTrans, perm=[0,2,1])
      state_t = tf.concat([init_state[None, None, :], state_t], 0)
      state_t = tf.transpose(tf.squeeze(state_t))
      state_p_t[i, :, :] = state_t
    return tf.convert_to_tensor(state_p_t)

  def run_segment_c(self, name, max_t_idx):
    n = self.components.loc[name].n
    probTe = self.state_distributions_to_t_idx(name, max_t_idx)
    # concentration after outlet does not contribute
    c = n * probTe[:, :-1, :] / self.seg_vols[None, :, None] #1 x |z| - 1 x |t|
    return tf.squeeze(c)

  #lowercase a for single particle A for specimen
  def experienced_concentration_aB(self, name_A, name_B, max_idx):
    # calculate c_B: run_segment_c includes t=0 so rm it
    c_B_seg = self.run_segment_c(name_B, max_idx)[None, ..., 1:-1]
    ell_A, ell_B = tuple(self.ellDi[c] for c in (name_A, name_B))
    # assume concentration at top plate negligible
    return c_B_seg / (ell_A + ell_B)[None, None, 1:max_idx] # remove focus cond

  def path_probs(self, name, rt_idx):
    rt_idx = np.atleast_1d(rt_idx)
    #|rt_idx| x |t| x |z|
    hiddenProbArr = self.state_prob_hidden(name, rt_idx).numpy()
    for i, end_t in enumerate(rt_idx):
      hiddenProbArr[i, end_t:, :] = 0. # probs until detection
    # omit intial state
    return tf.convert_to_tensor(hiddenProbArr[:, 1:, :])

  def path_experienced_concentration_avg(self, name_A, name_B, rt_idx):
    c_aB_exp = self.experienced_concentration_aB(name_A, name_B, np.max(rt_idx))
    c_aB_exp = tf.cast(c_aB_exp, tf.float32)
    #|rt_idx| x |t| x |z|
    hiddenProbTe = self.path_probs(name_B, rt_idx)
    weighted_c = tfm.multiply(tf.transpose(c_aB_exp, [0, 2, 1]), hiddenProbTe[..., :-1])
    avg_c = tfm.reduce_sum(weighted_c, axis=[1, 2]) / weighted_c.shape[1]
    return avg_c

  def portion_exp_c_threshold(self, c_th, name_A, name_B, rt_idx):
    c_th = np.atleast_1d(c_th)
    c_aB_exp = self.experienced_concentration_aB(name_A, name_B, np.max(rt_idx))
    #|rt_idx| x |t| x |z|
    hiddenProbTe = self.path_probs(name_A, rt_idx)
    # |c_th| x |rt_idx| x |t| x |z|
    selection = c_aB_exp[None, ...] > c_th[:, None, None, None]
    props_above_th = hiddenProbTe[selection]
    return props_above_th

  def state_prob_hidden(self, name, rt_idx):
    init_state = self.init_states[name]
    rt_idx = np.atleast_1d(rt_idx)
    len_rts = len(rt_idx)
    n_component = np.sum(init_state)
    init_probs = init_state / n_component
    init_distribution = tfd.Categorical(probs=tf.cast(init_probs, tf.float32),
                                        dtype=tf.int32)
    transTensor = np.swapaxes(self.transTensorDi[name], 1, 0)
    # only homogeneous transitions allowed (isocratic)
    transition_distribution = tfd.Categorical(
        probs=transTensor[:, ..., 0].astype('float32'), name='trans')
    observation_distribution = tfd.Empirical(
      np.append(self.z_centers, self.z_lims[-1]).astype('float32')[:, None])
    # TODO make outlet segment dummy rather than z_lims[-2]
    model_steps = np.max(rt_idx).astype('int32')

    model = tfd.HiddenMarkovModel(
        initial_distribution=init_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=model_steps)  # initial state is included

    #only concerned with time with index rt_ind (detection time)
    mask = np.ones((len_rts, model_steps), dtype=int)
    unmasked = np.zeros_like(mask, dtype=int)
    unmasked[np.arange(len_rts), rt_idx - 1] = 1
    mask = (mask - unmasked).astype('bool')
    obs_mold = tf.ones((len_rts, model_steps), dtype=tf.float32)
    observations = self.z_lims[-1] * obs_mold
    probs = model.posterior_marginals(observations=observations, mask=mask)
    return probs.probs_parameter()
