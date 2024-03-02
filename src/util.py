import read_vmec
import yaml

def read_vmec_file(filename):

    return read_vmec.vmec_data(filename)

def read_yaml(filename):

    with open(filename) as yaml_file:
        return yaml.safe_load(yaml_file)
    
