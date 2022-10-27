import parametric_stellarator


plas_eq = 'wout_daz.nc'
radial_build = {
    'sol': 10,
    'first_wall': 5,
    'blanket': 50,
    'back_wall': 5,
    'shield': 30,
    'coolant_manifolds': 30,
    'gap': 20,
    'vacuum_vessel': 30
    }
num_periods = 4

parametric_stellarator.parametric_stellarator(plas_eq, radial_build, num_periods)
