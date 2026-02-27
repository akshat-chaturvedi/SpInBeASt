import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from star_props import *
from cmcrameri import cm
import seaborn

m_p = []
m_c = []
t_p = []
t_c = []
l_p = []
l_c = []
star_names = []
for key in stellar_properties.keys():
    m_p.append(stellar_properties[f"{key}"]["m_p"])
    m_c.append(stellar_properties[f"{key}"]["m_c"])
    t_p.append(stellar_properties[f"{key}"]["T_p"])
    t_c.append(stellar_properties[f"{key}"]["T_s"])
    l_p.append(stellar_properties[f"{key}"]["L_p"])
    l_c.append(stellar_properties[f"{key}"]["L_s"])
    star_names.append(stellar_properties[f"{key}"]["name"])

m_p = np.array(m_p)

plt.rcParams['font.family'] = 'Trebuchet MS'
fig, ax = plt.subplots(figsize=(20, 10))
scatter1 = ax.scatter(t_p, l_p, marker="o", s=150, zorder=2, c=m_p, cmap="rocket", edgecolors="k", linewidth=2, label="Be")
scatter2 = ax.scatter(t_c, l_c, marker="d", s=150, zorder=2, c=m_p, cmap="rocket", edgecolors="k", linewidth=2, label="Companion")
# scatter1 = ax.scatter(t_p, l_p, edgecolor="k", marker="o", s=100, facecolor='none', zorder=2)
# scatter2 = ax.scatter(t_c, l_c, color="k", marker="o", s=100, zorder=2)
ax.plot([t_p, t_c], [l_p, l_c], zorder=0, c="k", alpha=0.5, linewidth=3)
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='both', which='major', length=10, width=1)
ax.set_xlabel(r"T$_{\text{eff}}$ [kK]", fontsize=22)
ax.set_ylabel(r"log(L [L$_{☉}$])", fontsize=22)
ax.set_xlim(8,90)
ax.grid(visible=True, alpha=0.5, linestyle="--")
ax.set_ylim(-0.5,4.6)
ax.legend(loc="lower left", fontsize=22)
ax.invert_xaxis()
cbar = fig.colorbar(scatter1, orientation='vertical', pad=0.01)
cbar.set_label(r'M$_{\text{Be}}$ [M$_{☉}$]', fontsize=22)
cbar.ax.tick_params(labelsize=18, length=8, width=1)

cursor = mplcursors.cursor(scatter2, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(star_names[sel.index]))

cursor1 = mplcursors.cursor(scatter1, hover=True)
cursor1.connect("add", lambda sel: sel.annotation.set_text(star_names[sel.index]))
ax.set_title("Be+sdOB Targets", fontsize=24)
# plt.show()
plt.savefig("Mass_Comp.pdf", dpi=300, bbox_inches="tight")