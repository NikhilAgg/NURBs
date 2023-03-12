import matplotlib.pyplot as plt


# # ro = 0.5	 ri = 0.25	 l = 1	 ep_list = 1 	 ep_step = 0.01	 lc = 0.1
# x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025000000000000005]
# y_points = [0, 2.6643321082225744e-05, 6.240687888244863e-05, 3.035131532470078e-05, 8.408674409403686e-05, 0.0001043606590893371]
# omegas = [5.999996626026256, 5.99979108862439, 5.9994698577014764, 5.999269732542086, 5.998983816390368, 5.998731361752424]
# delta_ws = [-0.02055374018663514, -0.03212309229132515, -0.02001251593908293, -0.028591615171791318, -0.025245463794387746]
# dw = (-0.023218072294857714+4.535552493816452e-19j)
# plt.plot(x_points, y_points, label = 0.1)


# # ro = 0.5	 ri = 0.25	 l = 1	 ep_list = 1 	 ep_step = 0.01	 lc = 0.09
# x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025000000000000005]
# y_points = [0, 1.8201669061300446e-05, 1.3668091596305565e-05, 8.214673318418107e-05, 0.00012742487214013052, 0.00010580525696247262]
# omegas = [6.001390986584804, 6.001139466233888, 6.000910681129499, 6.0006088838060565, 6.000585136729526, 6.000330198432494]
# delta_ws = [-0.025152035091569047, -0.022878510438939514, -0.03017973234422655, -0.002374707653007846, -0.025493829703204796]
# dw = (-0.023331868185439003-8.663307613306925e-20j)
# plt.plot(x_points, y_points, label = 0.09)


# # ro = 0.5	 ri = 0.25	 l = 1	 ep_list = 1 	 ep_step = 0.01	 lc = 0.08
# x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025000000000000005]
# y_points = [0, 5.3245886681324036e-05, 2.590920708423931e-05, 1.8596070136004924e-05, 1.244543357172619e-05, 8.846122107966821e-06]
# omegas = [6.002765974665234, 6.002584764320359, 6.002322971409205, 6.0020812020407, 6.001840595172579, 6.001602539629559]
# delta_ws = [-0.01812103448752822, -0.026179291115369097, -0.024176936850484054, -0.024060686812088505, -0.02380555430203657]
# dw = (-0.023445623155660622+4.9231399804208124e-20j)
# plt.plot(x_points, y_points, label = 0.08)


# ro = 0.5	 ri = 0.25	 l = 1	 ep_list = 1 	 ep_step = 0.01	 lc = 0.07
# x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025000000000000005]
# y_points = [0, 3.2169859021508545e-05, 2.4386296963919594e-05, 4.672885479306153e-05, 3.2793571976898824e-05, 2.9028808073125294e-05]
# omegas = [6.003671631926952, 6.003400747434813, 6.003169816363752, 6.002908759172805, 6.002683979822503, 6.002449029953289]
# delta_ws = [-0.02708844921395226, -0.02309310710604251, -0.026105719094715596, -0.022477935030185137, -0.023494986921424044]
# dw = (-0.123871463311801403-3.5265510753449313e-19j)
# plt.plot(x_points, y_points, label = 0.07)


# # ro = 0.5	 ri = 0.25	 l = 1	 ep_list = 1 	 ep_step = 0.01	 lc = 0.06
# x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025000000000000005]
# y_points = [0, 3.386785565233987e-05, 5.327507475870322e-05, 7.577529477734645e-05, 6.133205439492746e-05, 3.561983445412101e-05]
# omegas = [6.004805941989003, 6.004531007609025, 6.004270533865592, 6.004006967121247, 6.0037803438373025, 6.003564989532917]
# delta_ws = [-0.027493437997883063, -0.026047374343285412, -0.026356674434513394, -0.022662328394407183, -0.021535430438568426]
# dw = (-0.024106652432649076+2.7885570364770995e-19j)
# plt.plot(x_points, y_points, label = 0.06)

x_points = [0, 0.0001, 0.0004, 0.0009, 0.0016, 0.0025000000000000005]
delta_ws = [-0.027493437997883063, -0.025847374343285412, -0.0248356674434513394, -0.022662328394407183, -0.021535430438568426]
omegas = [6.003671631926952]
for i, d in enumerate(delta_ws):
    omegas.append(omegas[i] + d*0.01)

y_points = []
dw = (-0.028106652432649076+2.7885570364770995e-19j)
for i, w in enumerate(omegas):
    y_points.append(abs(w - (omegas[0] + dw * i*0.01)))

plt.plot(x_points, y_points, label = 0.06)
plt.show()