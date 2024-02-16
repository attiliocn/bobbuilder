def find_neighbors(adjacency_matrix, atom_number, visited=None, excluded_atoms=None, recursive=True):
    """0-indexed"""
    
    if visited is None:
        visited = set()
    if excluded_atoms is None:
        excluded_atoms = set()

    # Mark the current atom as visited
    visited.add(atom_number)

    # Get the neighbors of the current atom
    neighbors = [i for i, val in enumerate(adjacency_matrix[atom_number]) if val == 1 and i not in visited]

    # Remove excluded atoms and their neighbors
    neighbors = [neighbor for neighbor in neighbors if neighbor not in excluded_atoms]

    # Recursively find neighbors of neighbors
    if recursive:
        for neighbor in neighbors:
            if neighbor not in visited:
                find_neighbors(adjacency_matrix, neighbor, visited, excluded_atoms)
    else:
        return neighbors

    return list(visited)

#TODO -> refactor this function!