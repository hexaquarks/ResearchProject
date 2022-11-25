import numpy as np
import random 
import numpy.typing as np_t
import tifffile as tiff

def compute_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def get_last_point(path: list[tuple[float, float]]) -> tuple[float, float]:
    return path[-1][0], path[-1][1]

def get_x_coordinates(path: list[tuple[float, float]]) -> list[float]:
    return list(list(zip(*path))[0])

def get_y_coordinates(path: list[tuple[float, float]]) -> list[float]:
    return list(list(zip(*path))[1])

def get_random_normal_direction() -> float:
    # a =  np.random.normal(0, 1 / 2) 
    # if a < 0 : a = a * -1
    return np.random.normal(0, 1 / 2) 

def is_point_within_bounds(
    pos: tuple[float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    x, y = pos[0], pos[1]
    return (
        bounds[0][0] <= x <= bounds[0][1] and
        bounds[1][0] <= y <= bounds[1][1]
    )

def increment_tuple_by_val(tuple_object: tuple[float, float], val: tuple[float, float]) -> tuple[float, float]:
    return tuple_object[0] + val[0], tuple_object[1] + val[1]

def change_direction(
    tuple_object: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[float, float]:
    return tuple_object[0] - direction[0], tuple_object[1] - direction[1]

def get_random_gray_shade() -> tuple[float, float ,float, float]:
    black, gray = 0, 155
    max_val = 255
    return (random.randint(black, gray) / max_val,) * 3 + (1, )

def export_images_to_tiff(images: list[np_t.NDArray[np.float32]]):
    for i, image in enumerate(images):
        filename = f'data/tiffImages/beads-{i + 1}-ch1.tiff'
        tiff.imsave(filename, image)

def export_images_to_text():
    a = np.matrix(tiff.imread('./beads-5-ch1.tiff'))
    with open('values.txt', 'wb') as f:
        for line in a:
            np.savetxt(f, line, fmt='%.2f')
            