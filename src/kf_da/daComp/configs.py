from dataclasses import dataclass
from kf_da.velInit.IC_init import IC_init #from SRC.vp_floats.vp_py_utils import choose_exponent_format, float_pos_range

@dataclass
class KF_Opts:
    Re: float
    n: int
    NDOF: int
    dt: float
    total_T: float
    min_samp_T: float
    t_skip: float

@dataclass
class Particle_Opts:
    St: float
    beta: float

@dataclass
class DA_Opts:
    sigma_y: float
    x__y_sigma: float
    m_dt: any
    n_particles_list: any
    NT_list: any
    part_opts: Particle_Opts
    PIC_seed_list: any
    num_opt_inits: int
    TIC_seed_list: any
    ic_init: IC_init
    optimizer_list: any
    vp_list: any
    crit_list: any
    IC_param_list: any
    T_list: any



@dataclass
class Opt_Config_2:
    search_method: str
    ls_method: str
    its: str

    def __repr__(self):
        return (f"{self.search_method}_{self.ls_method}-{self.its}")


class VP_Float_Settings:
    def __init__(self, mbits, minv, maxv):
        exp_bits, exp_bias = choose_exponent_format(minv, maxv)
        self.minp, self.maxp = float_pos_range(exp_bits, exp_bias, mbits)

        self.mbits = mbits
        self.exp_bits = exp_bits
        self.exp_bias = exp_bias

    def get_vp_settings(self):
        return self.mbits, self.exp_bits, self.exp_bias
    
    def __repr__(self):
        return f"M={self.mbits}_E={self.exp_bits}_bias={self.exp_bias}"
