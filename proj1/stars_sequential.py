import math
import operator

from pandas import np

G = 1


def calculate_ri_rj(star_a, star_b):
    # print(f'{rank} star_a {star_a}', flush=True)
    # print(f'{rank} star_b {star_b}', flush=True)
    ri_rj = math.sqrt((star_a[1][0] - star_b[1][0]) ** 2 +
                      (star_a[1][1] - star_b[1][1]) ** 2 +
                      (star_a[1][2] - star_b[1][2]) ** 2)
    # print(f'ri_rj {ri_rj}', flush=True)
    return ri_rj


def calculate_interactions(old_ac, star_a, star_b):
    import operator
    # |ri-rj|
    ri_rj = calculate_ri_rj(star_a, star_b)
    # G * Mj/(|rj-ri|^3)*(xj-xi)
    ax = G * star_b[0] / ri_rj ** 3 * (star_b[1][0] - star_a[1][0])
    ay = G * star_b[0] / ri_rj ** 3 * (star_b[1][1] - star_a[1][1])
    az = G * star_b[0] / ri_rj ** 3 * (star_b[1][2] - star_a[1][2])
    new_ac = tuple(map(operator.add, old_ac, (ax, ay, az)))
    return new_ac


stars = [[100000, (1, 1, 1), (1, 1, 1)],
         [1, (2, 2, 2), (2, 1000, 2)],
         [100000, (3, 3, 3), (3, 3, 3)],
         [1, (4, 4, 4), (4, 1000, 4)]]

accu = np.zeros((4), dtype=[('ax', np.double), ('ay', np.double), ('az', np.double)])
print(f"accu: {accu}", flush=True)

for i in range(4):
    for j in range(4):
        if i == j:
            continue
        accu[i] = calculate_interactions(accu[i], stars[i], stars[j])

print(accu)

acc_t1_parallel = [(4.81146608e+03, 4.81146608e+03, 4.81146608e+03),
                   (4.81125224e-02, 4.81125224e-02, 4.81125224e-02),
                   (-4.81125224e+03, -4.81125224e+03, -4.81125224e+03),
                   (-2.13833914e+04, -2.13833914e+04, -2.13833914e+04)]

print(acc_t1_parallel)
