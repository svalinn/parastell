import parametric_stellarator


# Define plasma equilibrium VMEC file
plas_eq = 'wout_daz.nc'
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
# Define number of periods in stellarator plasma
num_periods = 4
# Define number of periods to generate
gen_periods = 4

# Create stellarator
parametric_stellarator.parametric_stellarator(
    plas_eq, num_periods, radial_build, gen_periods,
    exclude = ['plasma', 'sol'], h5m_export = 'cubit', plas_h5m_tag = 'Vacuum',
    facet_tol = 1e-2, len_tol = 1
    )
