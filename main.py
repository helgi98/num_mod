import matplotlib.pyplot as plt

from calculator import HUCalculator

mu = lambda x: 1
beta = lambda x: 50 * pow(x, 2)
sigma = lambda x: 40 * pow(x, 2)
f = lambda x: 40 * pow(x - 0.3, 5)

N = 5
a = -1
b = 1

calc = HUCalculator(N, a, b, mu, beta, sigma, f)
calc = calc.calc_until(5, 50)

print("N\t| Норма оцінювача\t\t| Норма розв'язку")
for i in range(len(calc.Ns)):
    print(f"{calc.Ns[i]}\t| {calc.eNs[i]}\t| {calc.uNs[i]}")

curr_pos = 0

fig, ax = plt.subplots()
ax.plot(calc.get_x(), calc.get_y(), marker='o')
plt.xlabel('x')
plt.ylabel(f'u')
plt.show()
