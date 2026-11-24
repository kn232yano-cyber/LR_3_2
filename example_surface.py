import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Еталонна функція y = (x1^2 - 8) * cos(x2)
n = 25
x1_grid = np.linspace(0, 4, n)
x2_grid = np.linspace(0, 4, n)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Y_target = (X1**2 - 8) * np.cos(X2)

# 2. Оголошення нечітких змінних (універсумів)
x1 = ctrl.Antecedent(np.linspace(0, 4, 101), 'x1')
x2 = ctrl.Antecedent(np.linspace(0, 4, 101), 'x2')
y = ctrl.Consequent(np.linspace(-10, 10, 201), 'y')

# 3. Функції належності (згідно з кроками 9-14 методички)
# x1: 3 трикутні терми (L, A, H)
x1['L'] = fuzz.trimf(x1.universe, [0, 0, 2])
x1['A'] = fuzz.trimf(x1.universe, [0, 2, 4])
x1['H'] = fuzz.trimf(x1.universe, [2, 4, 4])

# x2: 5 гаусових термів (L, LA, A, HA, H)
sigma = 0.42
x2['L'] = fuzz.gaussmf(x2.universe, 0, sigma)
x2['LA'] = fuzz.gaussmf(x2.universe, 1, sigma)
x2['A'] = fuzz.gaussmf(x2.universe, 2, sigma)
x2['HA'] = fuzz.gaussmf(x2.universe, 3, sigma)
x2['H'] = fuzz.gaussmf(x2.universe, 4, sigma)

# y: 5 трикутних термів (L, LA, A, HA, H)
y['L'] = fuzz.trimf(y.universe, [-10, -10, -5])
y['LA'] = fuzz.trimf(y.universe, [-10, -5, 0])
y['A'] = fuzz.trimf(y.universe, [-5, 0, 5])
y['HA'] = fuzz.trimf(y.universe, [0, 5, 10])
y['H'] = fuzz.trimf(y.universe, [5, 10, 10])

# Візуалізація функцій належності (можна зберегти для звіту)
x1.view()
plt.title('Функції належності змінної x1')
x2.view()
plt.title('Функції належності змінної x2 (gaussmf)')
y.view()
plt.title('Функції належності вихідної змінної y')

# 4. База з 9 правил (крок 16)
rules = [
    ctrl.Rule(x1['L'] & x2['L'], y['L']),
    ctrl.Rule(x1['L'] & x2['H'], y['A']),
    ctrl.Rule(x1['L'] & x2['HA'], y['H']),
    ctrl.Rule(x1['H'] & x2['L'], y['HA']),
    ctrl.Rule(x1['H'] & x2['H'], y['L']),
    ctrl.Rule(x1['A'] & x2['A'], y['A']),
    ctrl.Rule(x1['A'] & x2['HA'], y['HA']),
    ctrl.Rule(x1['L'] & x2['LA'], y['LA']),
    ctrl.Rule(x1['A'] & x2['HA'], y['A'])
]

# 5. Створення системи керування та симулятора
system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# Перевірка в точці [2, 2] (як на рис. 6 методички)
sim.input['x1'] = 2.0
sim.input['x2'] = 2.0
sim.compute()
print(f"Результат нечіткого виводу при x1=2, x2=2: y = {sim.output['y']:.4f}")
print(f"Точне математичне значення при x1=2, x2=2: y = {(2**2 - 8) * np.cos(2):.4f}")
y.view(sim=sim)
plt.title('Акумуляція та дефазифікація при x1=2, x2=2')

# Розрахунок 3D-поверхні нечіткого виводу
Y_fuzzy = np.zeros_like(Y_target)
for i in range(n):
    for j in range(n):
        sim.input['x1'] = X1[i, j]
        sim.input['x2'] = X2[i, j]
        sim.compute()
        Y_fuzzy[i, j] = sim.output['y']

# 6. Порівняльний 3D-графік: Еталон vs Нечітка система
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, Y_target, cmap='viridis')
ax1.set_title('Target: y = (x1^2 - 8)*cos(x2)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1, X2, Y_fuzzy, cmap='plasma')
ax2.set_title('Fuzzy Surface (СНВ Мамдані)')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

plt.tight_layout()
plt.show()