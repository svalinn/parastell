import parametric_stellarator


# Define plasma equilibrium VMEC file
plas_eq = 'wout_daz.nc'
# Define number of periods in stellarator plasma
num_periods = 4
# Define radial build
radial_build = {
'sol': {'thickness': 10, 'h5m_tag': 'Vacuum'},
'first_wall': {'thickness': 5},
'blanket': {'thickness': 50},
'back_wall': {'thickness': 5},
'shield': {'thickness': 30},
'coolant_manifolds': {'thickness': 30},
'gap': {'thickness': 20, 'h5m_tag': 'Vacuum'},
'vacuum_vessel': {'thickness': 30}
}
# Define number of periods to generate
gen_periods = 4
# Define magnet coil parameters
magnets = {
        'file': 'coils.wistd_306_best',
        'cross_section': ['circle', 20],
        'start': 3,
        'stop': None,
        'name': 'magnet_coils',
        'h5m_tag': 'magnets'
        }

# Create stellarator
parametric_stellarator.parametric_stellarator(
    plas_eq, num_periods, radial_build, gen_periods,
    exclude = ['plasma', 'sol'], plas_h5m_tag = 'Vacuum',
    include_magnets = True, magnets = magnets,
    h5m_export = 'cubit', facet_tol = 1, len_tol = 5
    )
