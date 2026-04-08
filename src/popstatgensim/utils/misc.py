"""Miscellaneous small helpers used across popstatgensim."""


def get_pop_kwargs(i, **kwargs) -> dict:
    '''
    Extracts population-specific parameters from SuperPopulation functions using kwargs.
    Parameters:
        i (int): Index of the population among active populations.
        **kwargs: Additional keyword arguments.
    Returns:
        pop_kwargs (dict): Dictionary of population parameters.
    '''
    pop_kwargs = {}
    # loops through each key-value pair in kwargs
    for key, value in kwargs.items():
        # if the value is a list, try to get the i-th element
        if isinstance(value, list):
            if i < len(value):
                pop_kwargs[key] = value[i]
            else:
                raise IndexError(f"Not enough elements in list for parameter '{key}' to match all populations.")
        # if not a list, use the value directly for all populations
        else:
            pop_kwargs[key] = value
    return pop_kwargs

def to_bits(n: int, bits: int):
    '''
    Converts integer n to list of bits of length `bits`.
    Parameters:
        n (int): Integer to convert to bits.
        bits (int): Number of bits to convert to.
    Returns:
        bit_list (list): List of bits representing integer n.
    '''
    return [int(b) for b in format(n, f'0{bits}b')]
