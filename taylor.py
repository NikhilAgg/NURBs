import numpy
import matplotlib.pyplot as plt

def func3(x):
    return 100*x**3 - 300*x**2 + 297*x

def func2(x):
    return -x**3

def func1(x):
    return -1.005*x**3

ep_step = 0.001
start = 1
deriv = -3

x = []
y1 = []
y2 = []
y3 = []
for i in range(10):
    epsilon = ep_step*i
    fd1 = func1(start + epsilon)
    fd2 = func2(start + epsilon)
    fd3 = func3(start + epsilon)
    ad1 = func1(start) + deriv*epsilon
    ad2 = func2(start) + deriv*epsilon
    ad3 = func3(start) + deriv*epsilon

    x.append(epsilon**2)
    y1.append(abs(fd1-ad1))
    y2.append(abs(fd2-ad2))
    y3.append(abs(fd3-ad3))

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(x, y1, label = "$\omega = -1.3x^3$")
plt.scatter(x, y1)
plt.plot(x, y2, label = "$\omega = -x^3$")
plt.scatter(x, y2)
plt.plot(x, y3, label = "$\omega = 100x^3 - 300x^2 + 297x$")
plt.scatter(x, y3)


plt.ylabel('$|\omega_{FD}\'-\omega\'|$')
plt.xlabel("$\epsilon^2$")
plt.legend()


plt.show()