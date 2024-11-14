import configparser;
import math;
import numpy as np;
import matplotlib as mpl
import matplotlib.pyplot as plt;
import scipy.linalg as linalg;

# # # -------------------------------------------- # # #
# # # load the variables from the config.ini file. # # #
# # # -------------------------------------------- # # #

config = configparser.ConfigParser()
config.read('config.ini')
b = int(config['DEFAULT']['b']) # branching parameter
m = int(config['DEFAULT']['m']) # layers of the graph approximation
j_max = int(config['DEFAULT']['j']) # the number of orthogonal polynomials to generate
ell = float(config['DEFAULT']['ell']) # height of the first branch of the bubble fractal
mode = str(config['DEFAULT']['display_mode']) # 2D or 3D

# # # ----------------------------------------- # # #
# # # constants for the bubble-diamond fractal. # # #
# # # ----------------------------------------- # # #

# # # renormalization constant # # #
r = b / (2 * b + 1)

# # # measure # # #
mu = 1 / (b + 2)

# # # harmonic extension algorithm # # #
lambda1 = (b + 1) / (2 * b + 1)
lambda2 = b / (2 * b + 1)

# # # Green's function # # #
alpha = b * (b + 1) / (2 * b + 1) / (2 * b + 1)
beta = b * b / (2 * b + 1) / (2 * b + 1)

# # # ------------------------------------------- # # #
# # # definitions for the bubble-diamond fractal. # # #
# # # ------------------------------------------- # # #

R = linalg.block_diag(*tuple([np.array([1/3]) for i in range(0, 2 * b + 4)]))
""" Stores the linear transformations associated with the given similitudes. 
    Should be stored as a block diagonal matrix of (2, 2) linear transformations containing all necessary scaling and rotations. 
"""

def B_func(i): return 1/3 if i % 2 == 0 else 0 if b == 1 else ell - 2 * ell * (i -1) / 2 / (b - 1)
B = np.array([0, 0] + [1/3 if i % 2 == 0 else 0 if b == 1 else ell - 2 * ell * (i -1) / 2 / (b - 1) for i in range(0, 2 * b)] + [2/3, 0]).reshape(-1, 1)
""" Stores the translations associated with the given similitudes.
    Should be stored as a stack of (2, 1) matrices containing all neccessary translations.
"""

H = np.vstack(tuple([[[1, 0], [lambda1, lambda2]]] + ([[lambda1, lambda2], [lambda2, lambda1]] * b)  + [[[lambda2, lambda1], [0, 1]]]))
""" Stores the matricies for the harmonic extension algorihm.
    Should be stored as a block diagonal matrix of (2, 2) harmonic extension matrices.
"""

G = np.array(*tuple([[[0 for i in range(0, 2 * b + 4)]] + [[0, alpha] + [alpha, beta] * b + [beta, 0]]  + ([[0, alpha] + [alpha, beta] * b + [beta, 0]] + [[0, beta] + [beta, alpha] * b + [alpha, 0]]) * b + [[0, beta] + [beta, alpha] * b + [alpha, 0]] + [[0 for i in range(0, 2 * b + 4)]]]))
def gamma(i1, i2, n1, n2):
    """Returns the coefficients for Green's function given two values of Fi(vn).

    Args:
        i1 (int): The index of the first similitude. 
        i2 (int): The index of the second similitude.
        n1 (int): The index of the first boundary vertex.
        n2 (int): The index of the second boundary vertex.

    Returns:
        float: The cooresponding coefficient for Green's function.
    """
    return G[2 * i1 + n1][2 * i2 + n2]

# # # ---------------------------- # # #
# # # construct the vertices of Gm # # #
# # # ---------------------------- # # #

V = np.array([[0, 1], [0, 0]])
""" Stores the vertices of Gm.
"""
for i in range(0, m): V = (R @ np.tile(V.reshape(-1), b + 2).reshape(2 * b + 4, -1) + B).reshape(-1, 2, V.shape[1]).swapaxes(0, 1).reshape(2, -1)

def warp(x, y, x0, y0, R):
    """Warps the vertices of the bubble graph for the equivalence relation.

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        x0 (float): x-coordiante of the center of the cell.
        y0 (float): y-coordinate of the center of the cell.
        R (float): Radius of the cell.

    Returns:
        float: Warped y-coordinate of the bubble grpah.
    """
    return (y - y0) * math.sqrt(round(1 - (x - x0) * (x - x0) / R / R, 10)) + y0
for level in range(1, round(math.log(V.shape[1] / 2, b + 2)) + 1):
    for i in range(0, round(V.shape[1] / 2 / (b + 2) ** level)):
        R = 1 / 3 ** (round(math.log(V.shape[1] / 2, b + 2)) + 1 - level) / 2
        x0 = V[0, 2 * (b + 2) ** level * i] + 3 * R
        y0 = V[1, 2 * (b + 2) ** level * i]
        for j in range(2 * (b + 2) ** (level - 1), 2 * (b + 2) ** level - 2 * (b + 2) ** (level - 1)):
            V[1, 2 * (b + 2) ** level * i + j] = warp(V[0, 2 * (b + 2) ** level * i + j], V[1, 2 * (b + 2) ** level * i + j], x0, y0, R)

if(mode == "3D"):

    # harmonic basis map to fill
    fb = []

    # monomial basis map to fill
    Pb = []

    # orthogonal polynomials map to fill
    ob = []

    # basis constants to fill
    a_b = [(b + 1) / (2 + 4 * b)]
    b_b = [b / (2 + 4 * b)]
    p_b = [(b + 1) / (2 * b + 1)]
    q_b = [b / (2 * b + 1)]

    # monomial constants to fill
    alpha_b = [1]
    beta_b = [1]
    eta_b = [0]
    gamma_b = [1]

    # construct the harmonic basis
    f00 = np.array([[1], [0]])
    f01 = np.array([[0], [1]])
    for i in range(0, m): f00 = (H @ f00).transpose().reshape(-1, 2).transpose()
    for i in range(0, m): f01 = (H @ f01).transpose().reshape(-1, 2).transpose()
    fb.append([f00, f01])

    # construct the first two monomials
    Pb.append([fb[0][0] + alpha_b[0] * fb[0][1], beta_b[0] * fb[0][1]])

    # function for computing the inner product of basis function
    def Ib_func(j, k, j_, k_):
        if((k == 0 and k_ == 0) or (k == 1 and k_ == 1)):   return a_b[j + j_]
        elif((k == 0 and k_ == 1) or (k == 1 and k_ == 0)): return b_b[j + j_]

    # function for computing the inner product of monomials
    def Ip_func(j, k, j_, k_):
        if(k == 0 and k_ == 0):   return sum(alpha_b[j - l] * eta_b[j_ + l + 1] - alpha_b[j_ + l + 1] * eta_b[j - l] for l in range(0, j + 1))
        elif(k == 0 and k_ == 1): return sum(alpha_b[j - l] * gamma_b[j_ + l + 1] - beta_b[j_ + l + 1] * eta_b[j - l] for l in range(0, j + 1))
        elif(k == 1 and k_ == 0): return sum(beta_b[j - l] * eta_b[j_ + l + 1] - alpha_b[j_ + l + 1] * gamma_b[j - l] for l in range(0, j + 1))
        elif(k == 1 and k_ == 1): return sum(beta_b[j - l] * gamma_b[j_ + l + 1] - beta_b[j_ + l + 1] * gamma_b[j - l] for l in range(0, j + 1))

    # function to find the value at a certain Fw values, with w input "backwards"
    def Fi(func, w):
        max_length = int(round(math.log(func.shape[1], b + 2), 10))
        offset0 = sum([w[i] * (b + 2) ** (max_length - i - 1) for i in range(0, len(w))])
        offset1 = sum([(b + 1) * (b + 2) ** (max_length - i - 1) for i in range(len(w), max_length)]) + offset0
        return [func[0, offset0], func[1, offset1]]
    
    # base conversion function to be used with Fi
    def numberToBase(n, b, d):
        if n == 0:
            return [0 for i in range(0, d)]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        while(len(digits) < d):
            digits.append(0)
        return digits[::-1]

    for j_level in range(1, 2 * j_max):

        # evaluate the p(j) and q(j) values
        p_b.append(-1 / (2 * b + 1) * (sum(p_b[j_level - l - 1] * ((b + 1) ** 2 * a_b[l] + b ** 2 * b_b[l]) + b * (b + 1) * q_b[j_level - l - 1] * (a_b[l] + b_b[l]) for l in range(0, j_level)) + (b + 1) * b_b[j_level - 1]))
        q_b.append(-1 / (2 * b + 1) * (sum(b * (b + 1) * p_b[j_level - l - 1] * (a_b[l] + b_b[l]) + q_b[j_level - l - 1] * ((b + 1) ** 2 * a_b[l] + b ** 2 * b_b[l]) for l in range(0, j_level)) + b * b_b[j_level - 1]))

        # evaluate the a(j) and b(j) values
        constant = (b + 2) ** (j_level + 1) * (2 * b + 1) ** (j_level + 1) / b ** j_level
        denom = -1 * (2 * b ** 3 + 4 * b ** 2 + 2 * b) / (2 * b + 1) * -1 * (2 * b ** 3 + 6 * b ** 2 + 6 * b + 2) / (2 * b + 1) - (constant - (2 * b ** 3 + 8 * b ** 2 + 7 * b + 2) / (2 * b + 1)) * (constant - (2 * b ** 3 + 6 * b ** 2 + 3 * b) / (2 * b + 1))
        a_b.append((-1 * (2 * b ** 3 + 6 * b ** 2 + 6 * b + 2) / (2 * b + 1) * (b + 1) * sum((a_b[l] + b_b[l]) * (b * p_b[j_level - l] + (b + 1) * q_b[j_level - l]) for l in range(0, j_level)) - (constant - (2 * b ** 3 + 6 * b ** 2 + 3 * b) / (2 * b + 1)) * (b + 1) * sum((a_b[l] + b_b[l]) * ((b + 1) * p_b[j_level - l] + b * q_b[j_level - l]) for l in range(0, j_level))) / denom)
        b_b.append((-1 * (2 * b ** 3 + 4 * b ** 2 + 2 * b) / (2 * b + 1) * (b + 1) * sum((a_b[l] + b_b[l]) * ((b + 1) * p_b[j_level - l] + b * q_b[j_level - l]) for l in range(0, j_level)) - (constant - (2 * b ** 3 + 8 * b ** 2 + 7 * b + 2) / (2 * b + 1)) * (b + 1) * sum((a_b[l] + b_b[l]) * (b * p_b[j_level - l] + (b + 1) * q_b[j_level - l]) for l in range(0, j_level))) / denom)

        # define the V1 values of the multiharmonic maps
        fj0 = np.zeros([2, b + 2])
        fj1 = np.zeros([2, b + 2])
        for n in range(0, 2):
            for i in range(0, b + 2):
                fj0[n][i] = -sum(mu * (r * mu) ** l * gamma(i, i_, n, n_) * Ib_func(l, k_, 0, n_) * Fi(fb[j_level - 1 - l][0], [i_])[k_] for i_ in range(0, b + 2) for n_ in range(0, 2) for k_ in range(0, 2) for l in range(0, j_level))
                fj1[n][i] = -sum(mu * (r * mu) ** l * gamma(i, i_, n, n_) * Ib_func(l, k_, 0, n_) * Fi(fb[j_level - 1 - l][1], [i_])[k_] for i_ in range(0, b + 2) for n_ in range(0, 2) for k_ in range(0, 2) for l in range(0, j_level))

        # define the rest of the values of the multiharmonic maps
        fb.append([fj0, fj1])
        while(fb[j_level][0].shape[1] < fb[0][0].shape[1]):
            tempj = [np.zeros([2, fb[j_level][0].shape[1] * (b + 2)]), np.zeros([2, fb[j_level][0].shape[1] * (b + 2)])]

            cur_length_log = round(math.log(fb[j_level][0].shape[1], b + 2))
            for w_index in range(0, fb[j_level][0].shape[1]):
                w = numberToBase(w_index, b + 2, cur_length_log)

                scaling_factor = b ** j_level / (2 * b + 1) ** j_level / (b + 2) ** j_level
                Fl0fw = [Fi(fb[l][0], w) for l in range(0, j_level + 1)]
                Fl1fw = [Fi(fb[l][1], w) for l in range(0, j_level + 1)]

                for n in range(0, 2): 
                    tempj[0][n, w_index] = (Fl0fw[j_level][n] + sum(p_b[j_level - l] * Fl1fw[l][n] for l in range(0, j_level + 1))) * scaling_factor
                    tempj[0][n, fb[j_level][0].shape[1] * (b + 1) + w_index] = sum(q_b[j_level - l] * Fl0fw[l][n] for l in range(0, j_level + 1)) * scaling_factor
                for n in range(0, 2): 
                    tempj[1][n, w_index] = sum(q_b[j_level - l] * Fl1fw[l][n] for l in range(0, j_level + 1)) * scaling_factor
                    tempj[1][n, fb[j_level][0].shape[1] * (b + 1) + w_index] = (Fl1fw[j_level][n] + sum(p_b[j_level - l] * Fl0fw[l][n] for l in range(0, j_level + 1))) * scaling_factor
                for i in range(1, b + 1):
                    for n in range(0, 2): tempj[0][n, fb[j_level][0].shape[1] * i + w_index] = sum(p_b[j_level - l] * Fl0fw[l][n] + q_b[j_level - l] * Fl1fw[l][n] for l in range(0, j_level + 1)) * scaling_factor
                    for n in range(0, 2): tempj[1][n, fb[j_level][1].shape[1] * i + w_index] = sum(q_b[j_level - l] * Fl0fw[l][n] + p_b[j_level - l] * Fl1fw[l][n] for l in range(0, j_level + 1)) * scaling_factor

            fb[j_level] = tempj
        
        # compute the values of the monomial constants
        if(j_level == 1):
            alpha_b.append(1/2)
        else:
            zeta = (b + 1) ** 2 / (b + 2) / (2 * b + 1) / ((b + 2) ** (j_level - 1) * (2 * b + 1) ** (j_level - 1) / b ** (j_level - 1) - 1)
            alpha_b.append(zeta * sum(alpha_b[j_level - l] * (2 * alpha_b[l] + sum(alpha_b[l - l_] * alpha_b[l_] for l_ in range(1, l + 1))) for l in range(1, j_level)))    
        iota = (b + 1) ** 2 / (2 * b + 1) / ((b + 2) ** j_level * (2 * b + 1) ** j_level / b ** j_level - 1)
        beta_b.append(iota * sum(beta_b[l] * sum(alpha_b[l_] * alpha_b[j_level - l - l_] for l_ in range(0, j_level - l + 1)) for l in range(0, j_level)))

        # construct the next monomials
        Pb.append([fb[j_level][0] + sum(alpha_b[j_level - l] * fb[l][1] for l in range(0, j_level + 1)), sum(beta_b[j_level - l] * fb[l][1] for l in range(0, j_level + 1))])

        # evaluate the inner product values
        eta_b.append(b_b[j_level - 1] + alpha_b[j_level] + sum(alpha_b[j_level - l] * a_b[l - 1] for l in range(1, j_level + 1)))
        gamma_b.append(beta_b[j_level] + sum(beta_b[j_level - l] * a_b[l - 1] for l in range(1, j_level + 1)))

    # computing the orthogonal polynomials inner products
    kj = []
    cjk = []
    dj = []
    for j in range(0, j_max):
        cjk.append([])
        for k in range(0, j): cjk[j].append((Ip_func(int(j / 2), j % 2, int(k / 2), k % 2) - sum(cjk[j][l] * cjk[k][l] / dj[l] for l in range(0, k))) * kj[k])
        oj = Pb[int(j / 2)][j % 2] - sum(cjk[j][l] / dj[l] * ob[l] for l in range(0, j))
        kj.append(1 / oj[oj.shape[0] - 1, 0])
        ob.append(oj * kj[j])
        dj.append((Ip_func(int(j / 2), j % 2, int(j / 2), j % 2) - sum(cjk[j][l] ** 2 / dj[l] for l in range(0, j))) * kj[j] ** 2)

    # reformat the functions out of jk notation into a list
    fb = np.array(fb).reshape(4 * j_max, 2, -1)
    Pb = np.array(Pb).reshape(4 * j_max, 2, -1)    

    # function to plot a function on the fractal
    def plot(ax, func, color_index_start, color_index_end):
        cmap = mpl.colormaps['magma']
        for i in range(0, int(V.shape[1] / 2)):
            xline = np.linspace(V[0, 2 * i], V[0, 2 * i + 1], 2)
            yline = np.linspace(V[1, 2 * i], V[1, 2 * i + 1], 2)
            zline = np.linspace(func[0][i], func[1][i], 2)
            color_index = color_index_start + (color_index_end - color_index_start) * xline[0]
            ax.plot(xline, yline, zline, color=cmap(color_index))

    # configure matplotlib settings
    mpl.rcParams['axes3d.xaxis.panecolor'] = (1, 1, 1)
    mpl.rcParams['axes3d.yaxis.panecolor'] = (1, 1, 1)
    mpl.rcParams['axes3d.zaxis.panecolor'] = (1, 1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.set_xticks(np.linspace(0, 1, 6, endpoint=True))
    ax.set_yticks(np.linspace(-1, 1, 3, endpoint=True))
    ax.set_zticks(np.linspace(-1, 1, 3, endpoint=True))

    # # # ------------------ # # #
    # # # functions to plot. # # #
    # # # ------------------ # # #

    for i in range(0, j_max):
        plot(ax, ob[i], i / j_max, (i + 1) / j_max)

    plt.show()

elif(mode == "2D"):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim([0, 1])
    plt.ylim([-0.5, 0.5])

    for i in range(0, int(V.shape[1] / 2)):
        xline = np.linspace(V[0, 2 * i], V[0, 2 * i + 1], 1000)
        yline = np.linspace(V[1, 2 * i], V[1, 2 * i + 1], 1000)
        ax.plot(xline, yline, color='black')

    plt.show()