import matplotlib.pyplot as plt

# ro = 0.5	 ri = 0.25	 l = 1	 ep_list = [1] 	 ep_step = 0.002	 lc = 0.005
# x_points = [0, 4e-06, 1.6e-05, 3.6e-05, 6.4e-05]
# y_points = [0, 3.9733904006779286e-07, 1.1091760287081087e-06, 2.134536541823981e-06, 3.4725708067187555e-06]
# omegas = [4.809691213543036, 4.809907830321822, 4.81012413260266, 4.810340121359973, 4.810555797443534]
# delta_ws = [0.10830838939313026, 0.108151140418844, 0.10799437865660622, 0.10783804178071676]
# delta_ws = [-0.0001572489742862615, -0.00015676176223777816, -0.00015633687588945122]
# dw = (0.10850705891316414+3.0764558135357634e-16j)
# plt.plot(x_points, y_points, label = 0.005)
# plt.scatter(x_points, y_points)

# # ro = 0.5	 ri = 0.25	 l = 1	 ep_list = [1] 	 ep_step = 0.0001	 lc = 0.005
# x_points = [0, 1e-08, 4e-08, 9.000000000000001e-08, 1.6e-07]
# y_points = [0, 1.2822661341494934e-08, 2.631824662178357e-08, 3.9983177328644444e-08, 5.476314942805396e-08]
# omegas = [4.809691213543036, 4.809702051426266, 4.809712888636572, 4.809723725677532, 4.8097345616034515]
# delta_ws = [0.1083788322997492, 0.10837210306036127, 0.10837040960609556, 0.10835925919217004]
# delta_ws = [-6.729239387937014e-06, -1.6934542657054408e-06, -1.1150413925520297e-05]
# dw = (0.10850705891316414+3.0764558135357634e-16j)
# plt.plot(x_points, y_points, label = 0.005)
# plt.scatter(x_points, y_points)


# ro = 0.5	 ri = 0.25	 l = 1	 ep_list = [1] 	 ep_step = 0.01	 lc = 0.005
x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016]
y_points = [0, 5.1218996013225465e-06, 1.801842445823057e-05, 3.845259865005923e-05, 6.657680422867546e-05]
omegas = [4.809691213543036, 4.810771162232566, 4.811843336296841, 4.8129079727117805, 4.8139649190953335]
delta_ws = [0.1079948689530319, 0.10721740642747335, 0.10646364149398124, 0.10569463835530257]
delta_ws = [-0.0007774625255585477, -0.000753764933492107, -0.00076900313867867]
dw = (0.10850705891316414+3.0764558135357634e-16j)
plt.plot(x_points, y_points, label = 0.005)
plt.scatter(x_points, y_points)

plt.show()


