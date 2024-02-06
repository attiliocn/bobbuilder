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

def reorder_xyz(xyz_matrix, src_numbering, dest_numbering):
            n_atoms = len(xyz_matrix)
            modified_matrix = np.zeros(n_atoms*4).reshape(n_atoms,4)
            modified_matrix = modified_matrix.astype(xyz_matrix.dtype)

            unmodified_in_source = []
            for i in range(n_atoms):
                if i not in src_numbering:
                    unmodified_in_source.append(i)
            unmodified_in_source = np.array(unmodified_in_source)
            
            empty_in_destination = []
            for i in range(n_atoms):
                if i not in dest_numbering:
                    empty_in_destination.append(i)
            empty_in_destination = np.array(empty_in_destination)

            for src, dest in zip(unmodified_in_source, empty_in_destination):
                modified_matrix[dest] = xyz_matrix[src]

            for src, dest in zip(src_numbering, dest_numbering):
                modified_matrix[dest] = xyz_matrix[src]
            
            return modified_matrix