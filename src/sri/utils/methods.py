def sum_vectors(vec1, vec2):
    """
    Adds two vectors represented as lists of tuples (index, value).

    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.

    Returns:
        list: Resultant vector after addition.
    """
    dict1 = {ind: valor for ind, valor in vec1}
    dict2 = {ind: valor for ind, valor in vec2}

    sum_dict = {ind: dict1.get(ind, 0) + dict2.get(ind, 0)
                for ind in set(dict1) | set(dict2)}

    sum_vec = [(ind, sum_dict[ind]) for ind in sum_dict]

    return sum_vec


def sub_vectors(vec1, vec2):
    """
    Subtracts two vectors represented as lists of tuples (index, value).

    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.

    Returns:
        list: Resultant vector after subtraction.
    """
    dict1 = {ind: valor for ind, valor in vec1}
    dict2 = {ind: valor for ind, valor in vec2}

    sub_dict = {ind: dict1.get(ind, 0) - dict2.get(ind, 0)
                for ind in set(dict1) | set(dict2)}

    sub_vec = [(ind, sub_dict[ind]) for ind in sub_dict]

    return sub_vec


def mult_scalar(vec, scalar):
    """
    Multiplies a vector by a scalar.

    Args:
        vec (list): Vector represented as a list of tuples (index, value).
        scalar (float): Scalar value to multiply the vector by.

    Returns:
        list: Resultant vector after scalar multiplication.
    """
    return [(ind, valor * scalar) for ind, valor in vec]


def mean(vectors):
    """
    Computes the mean vector from a list of vectors.

    Args:
        vectors (list): List of vectors, each represented as a list of tuples (index, value).

    Returns:
        list: Mean vector.
    """
    if len(vectors) == 0:
        return []

    r = []

    for vec in vectors:
        r = sum_vectors(r, vec)

    return mult_scalar(r, 1/len(vectors))
