import time


def timing(func):
    def wrapper(*args, **kwargs):
        if "numba" in func.__name__:
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            print(f"Kernel took {end - start} seconds (including compilation)")
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            print(f"Kernel took {end - start} seconds (excluding compilation)")
        else:
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds")
        return ret
    return wrapper