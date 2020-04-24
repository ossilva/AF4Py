# -*- coding: utf-8 -*-
import pandas as pd
from scipy.interpolate import CubicSpline

from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import solve_ivp

# compatibility with console and module
parent = Path(__file__).parent if \
'__file__' in vars() or '__file__' in globals() else Path('.')

def initialize_spline(table, interpolated_var):
  df = pd.DataFrame(table)
  y = df[interpolated_var]
  x = df.iloc[:, 0]
  return CubicSpline(x, y)

def get_t_idx(t, t_bins):
  # self.time_bins = t #np.concatenate((t, np.array([np.inf])))
  #returns 0 or len(bins) if outside
  idx = np.digitize(t, bins=t_bins, right=False) - 1
  if idx < 0 or t_bins[-1] < t:
    raise Exception(str(t) + ' is not within provided limits')
  return idx


def flow_at_seg(t, ts, col):
  # ml/min -> mm^3/s
  return 1000. * col[get_t_idx(t, ts)] / 60.

def centers(lims):
  return np.convolve(lims, np.ones(2), 'valid') / 2
  
def flow_segment_interp(flow_program, time_lims):
    """Calculates average flows for time segments from an emdf type program specs.

    :param flow_program: should contain ['deltat', 'focus', 'inject', 'xend', 'xstart', 'DetectorFlow'']
    :param segments: an array like of time bins 
    :returns: 
    :rtype: 

    """
    def vol(t, program_bins, vols, flow_fun, kind):
        def vol_offset(t_bin):
            if kind == 'cross':
                return (flow_fun(t) + flow_fun(t_bin)) * (t - t_bin) / 2
            if kind == 'inject':
                return flow_fun(t) * (t - t_bin)
        # handles digitize returning max idx + 1
        idx = vols.size - 1 if t == program_bins.iloc[-1] \
          else np.digitize(t, program_bins)
        vol_bin = vols.iloc[idx]
        return vol_bin + vol_offset(program_bins.iloc[idx])

    interpDF = flow_program.copy()
    interpDF = interpDF.append(
        pd.Series(np.zeros(interpDF.shape[1]), index=interpDF.columns), ignore_index=True)
    interpDF['xstart'].iloc[-1] = interpDF['xend'].iloc[-2]
    interpDF['t'] = interpDF['deltat'].cumsum().shift(1, fill_value=0.)
    interpDF['xavg'] = (interpDF['xstart'] + interpDF['xend']) / 2
    interpDF['xvol'] = (
        interpDF['xavg'] * interpDF['deltat']).cumsum().shift(1, fill_value=0.)
    interpDF['injvol'] = (
        interpDF['inject'] * interpDF['deltat']).cumsum().shift(1, fill_value=0.)

    t = interpDF['t']
    if time_lims[0] < t.iloc[0] or time_lims[-1] > t.iloc[-1]:
      raise Exception('Provided timestep limits outside program time')
    # only inject and cross flow are of interest assuming xend_i == xstart_i+1
    flowInterp = [
        interp1d(t, interpDF[kind], kind={
            'xstart': 'linear',
            'inject': 'previous'
        }[kind]) for kind in ['xstart', 'inject']
    ]
    FLOW_NAMES = ['cross', 'inject']
    flowInterp = dict(zip(FLOW_NAMES, flowInterp))
    program_bins = t
    vols = dict(zip(
        FLOW_NAMES,
        [interpDF['xvol'], interpDF['injvol']]
    ))

    avg_flow = {'cross': [], 'inject': []}
    del_t = np.diff(time_lims)
    for i, t in enumerate(time_lims[:-1]):
        for kind in avg_flow.keys():
            vol_a = vol(t, program_bins, vols[kind], flowInterp[kind], kind)
            vol_b = vol(
                time_lims[i+1], program_bins, vols[kind], flowInterp[kind], kind)
            avg_flow[kind].append((vol_b - vol_a) / del_t[i])

    # should not vary
    det_flow = flow_program['DetectorFlow'].iloc[
        np.digitize(time_lims[:-1], program_bins) - 1]

    avg_flow['detector'] = det_flow
    avg_flow['time'] = time_lims[:-1]
    avg_flow['del_t'] = del_t
    #index is left limit of timestep
    df = pd.DataFrame(data=avg_flow).set_index('time')
    df.eval('inlet = detector + cross - inject', inplace=True)
    df.index *= 60.
    return df.loc[:, ['inlet', 'cross', 'detector', 'inject', 'del_t']]

def convert_to_simple_program(flow_program):
  """Converts emdf type program to simple time dependent flow array

  :param flow_program: emdf type flow program
  :returns: array with columns {0: 't', 1: 'inlet', 2: 'cross'}
  :rtype: numpy.array

  """
  interpDF = flow_program.copy()
  t_lims = interpDF['deltat'].cumsum().to_numpy()
  t_lims = np.insert(t_lims, 0, 0.)
  df = flow_segment_interp(flow_program, t_lims).reset_index()
  end_lim = df.iloc[-1, :].copy()
  end_lim['time'] = t_lims[-1] + df['time'].iloc[-1]
  df = df.append(end_lim)
  return df.loc[:, ['time', 'inlet', 'cross']].to_numpy()

class Configuration():
  PORT_CYL_R = 0.75 # mm**2 src: outer diameter of fitting 1/16'' wyatt catalogue

  # trapezoidal channel geometry
  def bz_general(self, z, b0, bL, L, z0, zL):
    if (z < z0):
      return (z/z0)*b0
    elif (z < (L - zL)):
      return b0 + ((z - z0)/((L - zL) - z0)) * (bL-b0)
    elif (z <= L):
      return bL + (z - (L - zL))/zL * (-bL)
    elif(z < 0 or z > L):
      raise Exception('z not in range [{}, {}]'.format(0, L))
  
  def Az_general(self, z, b0, bL, L, z0, zL):
    _bz = partial(self.bz_general, b0=b0, bL=bL, L=L, z0=z0, zL=zL)
    _Az = partial(self.Az_general, b0=b0, bL=bL, L=L, z0=z0, zL=zL)
    if (z <= z0):
      return _bz(z) * z / 2.
    if (z <= (L - zL)):
      return _Az(z0) + (_bz(z0) + _bz(z))/2 * (z - z0)
    if (z <= L):
      lim = L - zL
      return _Az(lim) + (z - lim) * (_bz(lim) + _bz(z))/2

  def _top_port_flow_end_As(self):
    end_zs = np.array([self.z_origin, self.L]) +\
      np.array([self.PORT_CYL_R, -self.PORT_CYL_R])
    end_As = [self.w * self.bz(z) for z in end_zs]
    return end_zs, end_As

  def segment_Vs(self, partition_limits):
    Az_veczd = np.vectorize(self.Az)
    return self.w * np.diff(Az_veczd(partition_limits))

  #mean flow rate at z = vd0 - v_c * A(z)/A_C
  #FIXME independent of w so should not be in args
  def vdot_general(self, z, w, Atot, vd0, vdc, Az, t):
    return vd0(t=t) - vdc(t=t)*Az(z)/Atot
  
  # mean eluent velocity at z (mm/smm/min -> mm/s)
  def s_general(self, z, z_out, w, vdot, bz, t):
    #account for walls and outlet
    vdot_zt = vdot(z=z, t=t)
    if z < self.end_zs[0]:
      z = self.end_zs[0]
    elif z > self.end_zs[1]:
      z = self.end_zs[1]
    denom = (w*bz(z)) if z < z_out and z > 0. else np.finfo(float).eps
    return vdot_zt/denom

  def s_sample_general(self, s, R, z, t, lam, alpha, D):
    return R(lam=lam(D=D, t=t), alpha=alpha) * s(z=z, t=t)

  def l_general(self, D, u, t):
    return D / u(t)

  def lam_general(self, D, u, w, t):
    u = u(t)
    return D / (u * w) if u > 2.*D/w else 0.5

  def var_general(self, w, u_0, s, lam, RChi, alpha, D, del_t, z, t):
    u = u_0(t)
    lam = lam(D=D, t=t)
    return RChi(lam=lam, alpha=alpha) * w**2 * s(z=z, t=t)**2 * del_t / D

  def eq_zone_width_general (self, w):
    """Longitudinal zone width at ell.
    Reference: Wahlund and Giddings, “Properties of an Asymmetrical Flow Field-Flow Fractionation Channel Having One Permeable Wall.”

    :param w: channel width
    :returns: zone thickness
    :rtype: float

    """
    return w**2. / 6.

  def R_lam(self, lam, alpha):
    return 6*alpha*(1 - alpha) + (1 - 2*alpha)**2 * self.R_lam_c(lam) 

  def RChi_lam(self, lam, alpha):
    return (1 - 2*alpha)**6 * self.RChi_lam_c(lam)

  def lam(self, D, t):
    return self.lam_general(D=D, t=t, u=self.u_0, w=self.w)

  def __init__(self, name):
    if name == 'Wyatt SC':
      #units in mm
      self.b0 = b0 = 22.
      self.bL = bL = 6.
      self.L  = L  = 265.
      self.z0 = z0 = 20.
      self.zL = zL = 5.
      self.w  = w  = 0.35

      self.tabular_noneq = table = pd.read_csv(parent/'noneq_table350.csv')

    elif name == 'Wyatt MC':
      #units in mm
      self.b0 = b0 = 21.5
      self.bL = bL = 3.
      self.L  = L  = 173.
      self.z0 = z0 = 20.
      self.zL = zL = 3.
      self.w  = w  = 0.35

      self.tabular_noneq = table = pd.read_csv(parent/'noneq_table350.csv')

    else:
      raise Exception('Channel configuration does not exist for provided setup')

    self.z_origin = 0.
    self.geom_w = w

    self.u_0 = lambda t: self.vdc(t) / self.Az(self.L)
    self.l = partial(self.l_general, u=self.u_0)
    # self.lam = partial(self.lam_general, u=self.u_0, w=self.w)
    self.R_lam_c = initialize_spline(table, 'R')
    self.RChi_lam_c = initialize_spline(table, 'RChi')

    self.R_lam = self.R_lam
    self.RChi_lam = self.RChi_lam

    self.bz = partial(self.bz_general, b0=b0, bL=bL, L=L, z0=z0, zL=zL)
    self.Az = partial(self.Az_general, b0=b0, bL=bL, L=L, z0=z0, zL=zL)

    self.end_zs, self.end_As = self._top_port_flow_end_As()
    self.Atot = self.Az(L)
    self.V0 = self.Atot * w
    self.geom_V0  = self.V0


    #flow along z needs to be recalculated for each w
  def set_experimental_w(self, w):
    self.w = w
    self.V0 = self.Az(self.L) * w
    self.end_zs, self.end_As = self._top_port_flow_end_As()
    self.set_program(self.program)


  def _get_vol_flow(self, _type, t):
    if _type == 'inlet':
      col = 1
    if _type == 'cross':
      col = 2
    # ml/min -> mm^3/s
    return 1000. * self.program[get_t_idx(t, self.t_bins), col] / 60.

  def vd0(self, t):
    return self._get_vol_flow('inlet', t)

  def vdc(self, t):
    return self._get_vol_flow('cross', t)

  def set_program(self, t_vd_in_c):
    """Set a flow program
    computed using w

    :param t_vd_in_c: 
    :returns: 
    :rtype: 

    """

    self.program = t_vd_in_c
    self.t_bins = self.program[:, 0]

    self.vdot = partial(
        self.vdot_general,
        w=self.w,
        Atot=self.Atot,
        vd0=self.vd0,
        vdc=self.vdc,
        Az=self.Az
    )

    self.s = partial(
        self.s_general,
        # z_out=self.L - self.zL,
        z_out=self.L,
        w=self.w,
        vdot=self.vdot,
        bz=self.bz
    )

    def s_sample(z, t, alpha, D):
      return self.R_lam(lam=self.lam(D=D, t=t), alpha=alpha) * self.s(z=z, t=t)

    self.s_sample = s_sample

    self.var_zt = partial(
        self.var_general,
        w=self.w,
        u_0=self.u_0,
        s=self.s,
        lam=self.lam,
        RChi=self.RChi_lam
    )

  def set_emdf_program(self, emdf_program):
    self.emdf_program = emdf_program
    self.set_program(convert_to_simple_program(emdf_program))

  def set_time_partitions(self, time_lims):
    self.time_lims = time_lims
    #emdf uses minutes
    df = flow_segment_interp(self.emdf_program, time_lims/60.)
    df.rename(columns={
        'inlet': 'V_in',
        'inject': 'V_inj',
        'cross': 'V_c',
        'detector': 'V_out',
        'del_t': 'del_t'}, inplace=True)
    # df = df.round({'inlet': 8, 'inject': 8, 'cross': 8, 'detector': 8})
    # arbitrary precission for identifying duplicate conds
    df = df.round(8)
    # TODO id duplicates but using pandas functionality
    df.sort_values(df.columns.to_list(), inplace=True)
    uniq = ~(df.drop('del_t', axis=1).duplicated())
    df['uniq'] = uniq
    df['id'] = df['uniq'].cumsum() - 1
    df.sort_index(inplace=True)
    self.seg_prog = df

    vd0_seg = lambda t: flow_at_seg(t, df.index.to_numpy(), df['V_in'].to_numpy())
    vdc_seg = lambda t: flow_at_seg(t, df.index.to_numpy(), df['V_c'].to_numpy())
    u_0_seg = lambda t: vdc_seg(t) / self.Az(self.L)
    lam_seg = partial(self.lam_general, u=u_0_seg, w=self.w)

    vdot_seg = partial(
        self.vdot_general,
        w=self.w,
        Atot=self.Atot,
        vd0=vd0_seg,
        vdc=vdc_seg,
        Az=self.Az
    )

    self.s_seg = partial(
            self.s_general,
            # z_out=self.L - self.zL,
            z_out=self.L,
            w=self.w,
            vdot=vdot_seg,
            bz=self.bz
    )

    def s_sample_seg(z, t, alpha, D):
        return self.s_sample_general(
            s=self.s_seg, R=self.R_lam, z=z, t=t, lam=lam_seg, alpha=alpha, D=D)

    self.s_sample_seg = s_sample_seg

    self.var_zt_seg = partial(
        self.var_general,
        w=self.w,
        u_0=u_0_seg,
        s=self.s_seg,
        lam=lam_seg,
        RChi=self.RChi_lam
     )

  def zt_linear_flow_props(self, z_lims, alpha, D, use_rk=True):
    """
    Returns tuple of 1) times and corresponding condition ids
      2) dataframe with unique flow conditions and corresponding sample velocity
    and sd values at the center of z_lims

    """
    z_centers = centers(z_lims)
    propDF = self.seg_prog[self.seg_prog.uniq].set_index('id')
    propDi = dict()
    for i, row in propDF.iterrows():
      t = row.name
      del_t_sec = row.del_t * 60.
      z_sdM = np.empty((len(z_centers), 3))
      for j, z in enumerate(z_centers):
        if use_rk:
          ode_sol = solve_ivp(
            lambda t, y: self.s_sample_seg(z=y[0], t=t, alpha=alpha, D=D) if y[0] < self.L else\
            self.s_sample_seg(z=self.L, t=t, alpha=alpha, D=D),
                      t_span=np.array([t, t + del_t_sec]),
            y0=np.atleast_1d(z))
          del_z = ode_sol['y'][0, -1] - z
        else:
          del_z = del_t_sec * self.s_sample_seg(z=z, t=t, alpha=alpha, D=D)
        z_sdM[j, :] = np.array([
                      z,
                      del_z,
                      self.var_zt_seg(z=z, t=t, alpha=alpha, D=D, del_t=del_t_sec),
        ])
      propDi[row.name] = z_sdM
    propDF['propM'] = propDF.index.map(propDi)
    t_id_dict = self.seg_prog[['id']].astype('int').to_dict()['id']
    return t_id_dict, propDF
