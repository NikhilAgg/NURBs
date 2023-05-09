import matplotlib.pyplot as plt

# x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
# y_points = [0, 5.127036212212354e-07, 1.2101137147817285e-06, 1.739894886931923e-06, 2.4151326995922235e-06, 3.129780112724463e-06]
# plt.plot(x_points, y_points, label = 1e-1)

# x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
# y_points = [0, 3.004626762093506e-07, 6.257432845725101e-07, 9.926763810609216e-07, 1.3971934410463168e-06, 1.845145189322201e-06]
# plt.plot(x_points, y_points, label = 5e-2)

# x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
# y_points = [0, 7.470977994342392e-08, 1.8874921541917533e-07, 3.4206321804895073e-07, 5.347411412463072e-07, 7.664955251775033e-07]
# plt.plot(x_points, y_points, label = 1e-2)

x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
y_points = [0, 4.734213546609133e-08, 1.340575493475937e-07, 2.6007058438626426e-07, 4.253373370346591e-07, 6.297978001823458e-07]
plt.plot(x_points, y_points, label = 5e-3)
plt.scatter(x_points, y_points)

plt.xlabel('$\epsilon^2$')
plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
plt.legend()
plt.show()
