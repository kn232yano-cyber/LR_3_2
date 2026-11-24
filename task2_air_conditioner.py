import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Оголошення вхідних та вихідних змінних
temp = ctrl.Antecedent(np.linspace(10, 35, 101), 'temp')
rate = ctrl.Antecedent(np.linspace(-5, 5, 101), 'rate')
ac_control = ctrl.Consequent(np.linspace(-90, 90, 181), 'ac_control')

# 2. Функції належності для температури в приміщенні [10, 35] °C
temp['very_cold'] = fuzz.trapmf(temp.universe, [10, 10, 14, 17])
temp['cold'] = fuzz.trimf(temp.universe, [15, 18.5, 21.5])
temp['normal'] = fuzz.trimf(temp.universe, [20, 22.5, 25])
temp['warm'] = fuzz.trimf(temp.universe, [23.5, 26.5, 29.5])
temp['very_warm'] = fuzz.trapmf(temp.universe, [28, 31, 35, 35])

# 3. Функції належності для швидкості зміни температури [-5, 5]
rate['negative'] = fuzz.trapmf(rate.universe, [-5, -5, -2.5, -0.2])
rate['zero'] = fuzz.trimf(rate.universe, [-1.5, 0, 1.5])
rate['positive'] = fuzz.trapmf(rate.universe, [0.2, 2.5, 5, 5])

# 4. Функції належності для регулятора кондиціонера [-90, 90]
ac_control['large_left'] = fuzz.trapmf(ac_control.universe, [-90, -90, -70, -45])
ac_control['small_left'] = fuzz.trimf(ac_control.universe, [-55, -30, -5])
ac_control['off'] = fuzz.trimf(ac_control.universe, [-12, 0, 12])
ac_control['small_right'] = fuzz.trimf(ac_control.universe, [5, 30, 55])
ac_control['large_right'] = fuzz.trapmf(ac_control.universe, [45, 70, 90, 90])

# Візуалізація функцій належності
temp.view()
rate.view()
ac_control.view()

# 5. База з 12 правил керування кондиціонером за методичкою
rules = [
    # 1. Дуже тепла і швидкість додатня -> холод (великий кут вліво)
    ctrl.Rule(temp['very_warm'] & rate['positive'], ac_control['large_left']),
    # 2. Дуже тепла і швидкість від'ємна -> холод (невеликий кут вліво)
    ctrl.Rule(temp['very_warm'] & rate['negative'], ac_control['small_left']),
    # 3. Тепла і швидкість додатня -> холод (великий кут вліво)
    ctrl.Rule(temp['warm'] & rate['positive'], ac_control['large_left']),
    # 4. Тепла і швидкість від'ємна -> вимкнути
    ctrl.Rule(temp['warm'] & rate['negative'], ac_control['off']),
    # 5. Дуже холодна і швидкість від'ємна -> тепло (великий кут вправо)
    ctrl.Rule(temp['very_cold'] & rate['negative'], ac_control['large_right']),
    # 6. Дуже холодна і швидкість додатня -> тепло (невеликий кут вправо)
    ctrl.Rule(temp['very_cold'] & rate['positive'], ac_control['small_right']),
    # 7. Холодна і швидкість від'ємна -> тепло (великий кут вправо)
    ctrl.Rule(temp['cold'] & rate['negative'], ac_control['large_right']),
    # 8. Холодна і швидкість додатня -> вимкнути
    ctrl.Rule(temp['cold'] & rate['positive'], ac_control['off']),
    # 9. Дуже тепла і швидкість 0 -> холод (великий кут вліво)
    ctrl.Rule(temp['very_warm'] & rate['zero'], ac_control['large_left']),
    # 10 (13). В нормі і швидкість додатня -> холод (невеликий кут вліво)
    ctrl.Rule(temp['normal'] & rate['positive'], ac_control['small_left']),
    # 11 (14). В нормі і швидкість від'ємна -> тепло (невеликий кут вправо)
    ctrl.Rule(temp['normal'] & rate['negative'], ac_control['small_right']),
    # 12 (15). В нормі і швидкість 0 -> вимкнути
    ctrl.Rule(temp['normal'] & rate['zero'], ac_control['off']),
]

# 6. Створення системи керування
ac_ctrl = ctrl.ControlSystem(rules)
ac_sim = ctrl.ControlSystemSimulation(ac_ctrl)

# 7. Тестування на 4 сценаріях
test_cases = [
    (32, 3),     # Сценарій 1: Спека (32°C) і температура ще росте (+3)
    (22.5, 0),   # Сценарій 2: Комфортна норма (22.5°C) без змін (0)
    (13, -3),    # Сценарій 3: Дуже холодно (13°C) і стає ще холодніше (-3)
    (26.5, -2)   # Сценарій 4: Тепло (26.5°C), але температура вже спадає (-2)
]

print("=== Тестування системи керування кондиціонером ===")
for t_val, r_val in test_cases:
    ac_sim.input['temp'] = t_val
    ac_sim.input['rate'] = r_val
    ac_sim.compute()
    out_val = ac_sim.output['ac_control']
    print(f"Темп: {t_val}°C, Швидкість: {r_val:+} -> Регулятор: {out_val:+.2f}°")

# Візуалізація результату для першого сценарію (32°C, швидкість +3)
ac_sim.input['temp'] = 32
ac_sim.input['rate'] = 3
ac_sim.compute()
ac_control.view(sim=ac_sim)

plt.show()