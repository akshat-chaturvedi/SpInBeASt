import numpy as np
import matplotlib.pyplot as plt
from Sandbox.angleAnnotator import AngleAnnotation as AA
from random import uniform
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def pa(ra, err_ra, dec, err_dec):
    """
    Get the position angle of a binary companion given its ΔRA and ΔDec relative to the primary (centered at (0,0))

    Parameters:
        ra (float): ΔRA of the companion
        err_ra (float): error in the ΔRA of the companion
        dec (float): ΔDec of the companion
        err_dec (float): error in the ΔDec of the companion

    Returns:
        The position angle of the companion with error
    """
    a = np.rad2deg(np.arctan2(dec, ra))
    tot_err = a * (err_ra / ra + err_dec / dec)
    pos_angle_err = (tot_err / (1+a**2))
    if (ra > 0) and (dec > 0):
        pos_angle = 90-a
        return pos_angle, pos_angle_err
    else:
        if abs(a-90) > 90:
            pos_angle = abs(a-90)
            return pos_angle, pos_angle_err
        else:
            pos_angle = 360-abs(a-90)
            return pos_angle, pos_angle_err

def confidence_ellipse(x, y, axs, n_std=3.0, cov=None, face_color='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
        x, y : array-like, shape (n, )
            Input data.

        axs : matplotlib.axes.Axes
            The Axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radii.

        cov : Covariance matrix

        face_color : str
            The face-color of the ellipse.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
        matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    if cov is None:
        cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = float(np.sqrt(1 + pearson))
    ell_radius_y = float(np.sqrt(1 - pearson))
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=face_color, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = float(np.sqrt(cov[0, 0]) * n_std)
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = float(np.sqrt(cov[1, 1]) * n_std)
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + axs.transData)
    return axs.add_patch(ellipse)


if __name__ == '__main__':
    # x1, y1 = (uniform(-5, 5), uniform(-5, 5))
    breakpoint()
    x1, y1 = (0.6632769393729222, -3.9214591152894935)
    # x1_err = 0.05
    # y1_err = 0.0414
    x1_err = np.array([[0.045], [0.040]])
    y1_err= np.array([[0.036], [0.032]])
    cov_mat1 = np.array([[0.0007182481852045394, -1.9330263136829296e-06], [-1.9330263136829296e-06, 0.0002354532771113061]])

    x2, y2 = (0.5008613643445381, -8.170206894408196)
    cov_mat2 = np.array(
        [[0.0010225726932301862, 4.975014717975709e-05], [4.975014717975709e-05, 0.0006648492071095658]])
    x2_err = np.array([[0.025], [0.034]])
    y2_err = np.array([[0.052], [0.061]])

    # x3, y3 = (0.53673, 0.90386)
    # x3_err = 0.05
    # y3_err = 0.0414
    # cov_mat3 = np.array([[2.5e-3, 1.1e-05], [1.1e-05, 1.7e-3]])

    # x3, y3 = (0.6263719360103367, -4.275172065859552)
    # x3_err = 0.05
    # y3_err = 0.0414
    # cov_mat3 = np.array([[0.0005741740429037826, -0.00014375211339726917],
    #                      [-0.00014375211339726917, 0.0005703072621380656]])

    x3, y3 = (-1.379, -3.971)
    x3_err = np.array([[0.030], [0.028]])
    y3_err = np.array([[0.036], [0.035]])
    cov_mat3 = np.array([[0.00023588541841400182, -9.229969565858467e-05],
                         [-9.229969565858467e-05, 0.0002318667990351338]])

    x4, y4 = (4.362, -7.346)
    x4_err = np.array([[0.029], [0.032]])
    y4_err = np.array([[0.017], [0.018]])
    cov_mat4 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                         [-5.672761774141208e-05, 0.00016870022188802155]])

    x5, y5 = (0.527, -8.227)
    x5_err = np.array([[0.048], [0.049]])
    y5_err = np.array([[0.041], [0.042]])
    cov_mat5 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                         [-5.672761774141208e-05, 0.00016870022188802155]])

    x6, y6 = (1.737, -6.181)
    x6_err = np.array([[0.023], [0.021]])
    y6_err = np.array([[0.019], [0.017]])
    cov_mat6 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                         [-5.672761774141208e-05, 0.00016870022188802155]])

    x7, y7 = (-4.422, -0.686)
    x7_err = np.array([[0.040], [0.057]])
    y7_err = np.array([[0.059], [0.051]])
    cov_mat7 = np.array([[0.0001588813250489883, -5.672761774141208e-05],
                         [-5.672761774141208e-05, 0.00016870022188802155]])

    my_pa = pa(x3, y3)
    print(my_pa)

    plt.rcParams['font.family'] = 'Geneva'
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(0, 0, color="k", marker="+", s=200)

    # ax.scatter(x1, y1, color="red", marker="x", s=100)
    # ax.errorbar(x1, y1, xerr=x1_err, yerr=y1_err, color="red", marker="x")
    #
    # ax.scatter(x2, y2, color="red", marker="x", s=100)
    # ax.errorbar(x2, y2, xerr=x2_err, yerr=y2_err, color="red", marker="x")

    ax.scatter(x3, y3, color="red", marker="x", s=100, label="MIRCX 2021-May-30")
    ax.errorbar(x3, y3, xerr=x3_err, yerr=y3_err, color="red", marker="x")

    ax.scatter(x4, y4, color="magenta", marker="x", s=100, label="MIRCX 2025-Jul-05")
    ax.errorbar(x4, y4, xerr=x4_err, yerr=y4_err, color="magenta", marker="x")

    ax.scatter(x5, y5, color="green", marker="x", s=100, label="MYSTIC 2025-Jul-05")
    ax.errorbar(x5, y5, xerr=x5_err, yerr=y5_err, color="green", marker="x")

    ax.scatter(x6, y6, color="cyan", marker="x", s=100, label="MIRCX 2021-Jul-04")
    ax.errorbar(x6, y6, xerr=x6_err, yerr=y6_err, color="cyan", marker="x")

    ax.scatter(x7, y7, color="dodgerblue", marker="x", s=100, label="MIRCX 2021-Jul-03")
    ax.errorbar(x7, y7, xerr=x7_err, yerr=y7_err, color="dodgerblue", marker="x")

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.xaxis.set_inverted(True)

    # confidence_ellipse(np.array([x1]), np.array([y1]), ax, n_std=3, cov=cov_mat1, edgecolor='green')
    # confidence_ellipse(np.array([x2]), np.array([y2]), ax, n_std=3, cov=cov_mat2, edgecolor='dodgerblue')

    confidence_ellipse(np.array([x3]), np.array([y3]), ax, n_std=3, cov=cov_mat3, edgecolor='k', zorder=0)
    confidence_ellipse(np.array([x4]), np.array([y4]), ax, n_std=3, cov=cov_mat4, edgecolor='k',zorder=0)
    confidence_ellipse(np.array([x5]), np.array([y5]), ax, n_std=3, cov=cov_mat5, edgecolor='k', zorder=0)
    confidence_ellipse(np.array([x6]), np.array([y6]), ax, n_std=3, cov=cov_mat6, edgecolor='k', zorder=0)

    ax.set_title(fr'60 Cyg Interferometric Orbit', fontsize=24)
    ax.set_xlabel(r"E $\leftarrow\, \Delta \alpha$ (mas)", fontsize=22)
    ax.set_ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)', fontsize=22)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='both', which='major', length=10, width=1)
    ax.yaxis.get_offset_text().set_size(20)
    ax.legend(fontsize=12, ncol=2)
    fig.savefig("ErrorEllipse.pdf", bbox_inches="tight", dpi=300)
