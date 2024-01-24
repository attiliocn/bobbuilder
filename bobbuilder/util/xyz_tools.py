import numpy as np

def build_xyz_file(elements, coordinates):
    elements = elements.reshape(-1,1)
    coordinates = coordinates.reshape(-1,3)
    xyz_content = []
    xyz_content.append(f"{str(len(coordinates))}\n")
    xyz_content.append('\n')
    for i in np.concatenate((elements, coordinates), axis=1):
        xyz_content.append("{: >3} {: >10} {: >10} {: >10}\n".format(*i))
    xyz_content = ''.join(xyz_content)
    return xyz_content
    
def obabel(input_string, input_format, output_format):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(input_format, output_format)
    obabel_mol = openbabel.OBMol()
    obConversion.ReadString(obabel_mol,input_string)
    return obConversion.WriteString(obabel_mol)