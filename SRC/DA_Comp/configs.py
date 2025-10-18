from dataclasses import dataclass

@dataclass
class KF_Opts:
    Re: float
    n: int
    NDOF: int
    dt: float
    T: float
    min_samp_T: float
    t_skip: float

@dataclass
class Particle_Opts:
    St: float
    beta: float

@dataclass
class DA_Opts:
    n_particles_list: any
    sampling_period_list: any
    part_opts: Particle_Opts
    num_particle_inits: int
    num_opt_inits: int
    num_seeds: int
    int_pert_range: tuple
    optimizer_list: any
    crit_list: any
    T_list: any



@dataclass
class Opt_Config_2:
    search_method: str
    ls_method: str
    its: str

    def __repr__(self):
        return (f"{self.search_method}_{self.ls_method}-{self.its}")

