import numpy as np
from scipy.integrate import dblquad
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Function definitions from the user's provided code

def integrand(phi, rho, R, z, z0, z_prime, A):
    if rho > 0:
        return A * np.cos(phi) / np.sqrt(rho**2 + R**2 + (z - z0 - z_prime)**2 - 2 * R * rho * np.cos(phi))
    else:
       return -A * np.cos(phi) / np.sqrt(rho**2 + R**2 + (z - z0 - z_prime)**2 + 2 * R * rho * np.cos(phi))



def simpson_integration(f, a, b, n, rho, R, z, z0, z_prime, A):
    if n % 2:
        raise ValueError("Number of subintervals 'n' must be even.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x, rho, R, z, z0, z_prime, A)
    S = h / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    return R*S

def calculate_B_z(rho, R1, R2, z, L, A):
    def integrand(phi_prime, z_prime, rho, R, z):
        denominator = np.sqrt(rho**2 + R**2 + (z - z_prime)**2 - 2*R*rho*np.cos(phi_prime))
        return np.cos(phi_prime) / denominator

    def integral_A(rho, R, z, L, A):
        result, _ = dblquad(integrand, 0, L, lambda _: 0, lambda _: 2*np.pi, args=(rho, R, z))
        return rho * A *R* result

    rho = np.abs(rho)
    A_derivative = derivative(lambda r: integral_A(r, R1, z, L, A)-integral_A(r, R2, z, L, A), rho, dx=1e-6)
    B_z = 1/rho * A_derivative if rho != 0 else 0
    return B_z

def fit_function(rho, rho_prime,z, z0, A, R1, R2, L, B0, a, b, n):
    r = rho - rho_prime
    integral_value = simpson_integration(integrand, a, b, n, r, R1, z, z0, L, A) - \
                     simpson_integration(integrand, a, b, n, r, R1, z, z0, 0, A) + \
                     simpson_integration(integrand, a, b, n, r, R2, z, z0, 0, A) - \
                     simpson_integration(integrand, a, b, n, r, R2, z, z0, L, A)
    return integral_value + B0


# Function to convert Cartesian coordinates to magnetic field components

def cartesian_magnetic_field(xyz_array, R1, R2, L, A, B0, a, b, n, z0):
    results = []

    for xyz in xyz_array:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        # Conversion from Cartesian to cylindrical coordinates
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        # Calculating B_rho (B_phi is zero due to symmetry)
        B_rho = fit_function(rho, 0, z, z0, A, R1, R2, L, B0, a, b, n)

        # Calculating B_z
        B_z_value = calculate_B_z(rho, R1, R2, z, L, A)

        # Converting B_rho back to Cartesian coordinates
        B_x = B_rho * np.cos(phi) 
        B_y = B_rho * np.sin(phi)

        results.append((B_x, B_y, B_z_value))

    return np.array(results)


# 示例用法
# xyz_array = np.array([[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]])
# results = cartesian_magnetic_field(xyz_array, R1, R2, L, A, B0, a, b, n, z0)

#def cartesian_magnetic_field(x, y, z, R1, R2, L, A, B0, a, b, n, z0):
    # 向量化的 rho 和 phi 计算
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # 向量化的 B_rho 和 B_z 计算
    B_rho = np.vectorize(fit_function)(rho, 0, z0, A, R1, R2, L, B0, a, b, n)
    B_z = np.vectorize(calculate_B_z)(rho, R1, R2, z, L, A)

    # 转换回笛卡尔坐标
    B_x = B_rho * np.cos(phi) 
    B_y = B_rho * np.sin(phi) 

    return B_x, B_y, B_z
def transform_coordinates(coordinates,isolator_xyz, group):
    """
    Transforms the given coordinates based on the specified group (A or B).

    :param coordinates: numpy array of shape (n, 3), where each row is a coordinate (x, y, z).
    :param group: string, either 'A' or 'B', specifying the group for transformation rules.
    :return: transformed numpy array.
    """

    # Isolator coordinate (common for both groups)
    #isolator_xyz = np.array([-159.475, -189.344,  184.97])
    #isolator_xyz = 0

    # Compute difference with the isolator coordinate
    diff = isolator_xyz - coordinates

    # Transform based on group
    if group == 'A':
        diff[:, 0]=-diff[:, 0]
        # For group A, reorder coordinates as zyx
        return diff[:, [2, 1, 0]]
    elif group == 'B':
        # For group B, reorder coordinates as zxy
        return diff[:, [2, 0, 1]]
    
    elif group == 'C':
        # For group C, reorder coordinates as zyx
        return diff[:, [2, 1, 0]]
    elif group == 'D':
        # For group D, reorder coordinates as zxy
        diff[:, 1]=-diff[:, 1]
        return diff[:, [2, 0, 1]]
    else:
        raise ValueError("Group must be either 'A' or 'B'.")

# 给定的参数
A = 181
R1 = 45.5 / 2
R2 = 12.5 / 2
L = 47
B0 = 0  # 假设的额外磁场常数，可以根据实际情况调整
a = 0   # 积分下限
b = 2*np.pi  # 积分上限
n = 100  # 积分分割的子区间数量
z0 = 0  # z0值可以根据实际情况调整
z_p=100
# 示例：计算某一笛卡尔坐标点的磁场分量
#x, y, z = 0, 20, L+100

#使用方法
#B_x, B_y, B_z = cartesian_magnetic_field(x, y, z, R1, R2, L, A, B0, a, b, n, z0)
A_xyz=np.array([[374.0,60.0, 231.213],[386.134,  -52.841,  231.207],[361.728,  -127.755,  262.627],[411.1,  -252.759,  231.207]])

B_xyz=np.array([[162.929,  -563.531,  262.627],[88.055,  -613.51,  230.485]])
result=[]
for i in np.linspace(-30,30,1):
    isolator_xyz=np.array([-159.475,-189.344+370,231.207])
    Agroup=transform_coordinates(A_xyz,isolator_xyz, 'A')
    Bgroup=transform_coordinates(B_xyz,isolator_xyz, 'B')
    Afield=cartesian_magnetic_field(Agroup, R1, R2, L, A, B0, a, b, n, z0)[:, [2, 1, 0]]
    Afield[:, 0]=-Afield[:, 0]
    Bfield=cartesian_magnetic_field(Bgroup, R1, R2, L, A, B0, a, b, n, z0)[:, [1,2,0]]
    result.append((Afield.sum(axis=0)+Bfield.sum(axis=0)).tolist())


print (Afield)
print (Bfield)
print (Afield.sum(axis=0))
print (Bfield.sum(axis=0))
print (Afield.sum(axis=0)+Bfield.sum(axis=0))
