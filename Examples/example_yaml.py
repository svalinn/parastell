import yaml
import numpy as np

import parastell.parastell as ps

# Load configuration from YAML file
config_file = 'config.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Extract stellarator configuration parameters
vmec_file = config['vmec_file']
invessel_build = config['invessel_build']
magnets = config['magnet_coils']
source = config['source_mesh']
dagmc_export = config['dagmc_export']

# Generate parametric stellarator model
stellarator = ps.Stellarator(vmec_file)
stellarator.construct_invessel_build(invessel_build)
stellarator.construct_magnets(magnets)
stellarator.construct_source_mesh(source)
stellarator.export_dagmc(dagmc_export)
