import matplotlib.pyplot as plt

x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
y_points = [0, 6.952618709631926e-05, 7.427870386299542e-05, 4.1214378096215824e-05, 0.00014412722692467276, 1.9132510521074646e-05]
plt.plot(x_points, y_points, label = 5e-1)

x_points = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
y_points = [4.2581831627904505e-05, 1.5470589287051663e-05, 3.623065770984551e-05, 3.9317214440895816e-05, 5.783003885164292e-05]
plt.plot(x_points, y_points, label = 3e-1)

x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
y_points = [0, 1.668606674794686e-05, 3.6591120615862095e-05, 7.410886764591631e-05, 6.626341489498051e-05, 9.00307454529211e-05]
plt.plot(x_points, y_points, label = 2.5e-2)

x_points = [0, 1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05]
y_points = [0, 1.76110512525995e-05, 4.0615739197140594e-05, 5.5102651470224516e-05, 7.65229533168908e-05, 9.539168359885532e-05]
plt.plot(x_points, y_points, label = 2e-2)


plt.xlabel('$\epsilon^2$')
plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
plt.legend()
plt.show()
