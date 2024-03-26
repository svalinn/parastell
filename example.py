import yaml
import parastell
import numpy as np

# Load configuration from YAML file
config_file = 'config.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Access radial_build configurations
radial_build = config['radial_build']
phi_list = radial_build['phi_list']
theta_list = radial_build['theta_list']
repeat = radial_build['repeated'] 
wall_s = radial_build['wall_s'] 

# Correctly access stellarator_parameters
stellarator_parameters = config['stellarator_parameters']
num_phi = stellarator_parameters['num_phi']
num_theta = stellarator_parameters['num_theta']

# Prepare the build dictionary according to your requirements
build = {
    'phi_list': phi_list,
    'theta_list': theta_list,
    'wall_s': wall_s,
    'radial_build': radial_build['components'],
    'components': {}
}

# Convert thickness_matrix in 'radial_build' components to numpy arrays
for component, details in radial_build['components'].items():
    if 'thickness_matrix' in details and isinstance(details['thickness_matrix'], list):
        build['components'][component] = {
            'thickness_matrix': np.array(details['thickness_matrix'])
        }
        if 'h5m_tag' in details:
            build['components'][component]['h5m_tag'] = details['h5m_tag']

# Access VMEC file, magnets, source, and export configurations
plas_eq = config['VMEC_file']
magnets = config['magnet_coils']
source = config['source_mesh_parameters']
export = config['export_parameters']

# Create stellarator using the configurations
strengths = parastell.parastell(
    plas_eq, build, repeat, num_phi, num_theta,
    magnets=magnets, source=source, export=export
)
#Output