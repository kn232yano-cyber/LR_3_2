import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Оголошення вхідних та вихідних змінних
temp = ctrl.Antecedent(np.linspace(0, 100, 101), 'temp')
pressure = ctrl.Antecedent(np.linspace(0, 10, 101), 'pressure')

hot_valve = ctrl.Consequent(np.linspace(-90, 90, 181), 'hot_valve')
cold_valve = ctrl.Consequent(np.linspace(-90, 90, 181), 'cold_valve')

# 2. Функції належності для температури (5 термів)
temp['cold'] = fuzz.trapmf(temp.universe, [0, 0, 15, 25])
temp['cool'] = fuzz.trimf(temp.universe, [15, 30, 42])
temp['warm'] = fuzz.trimf(temp.universe, [35, 45, 55])
temp['not_very_hot'] = fuzz.trimf(temp.universe, [50, 62, 75])
temp['hot'] = fuzz.trapmf(temp.universe, [68, 80, 100, 100])

# 3. Функції належності для напору (3 терми)
pressure['weak'] = fuzz.trapmf(pressure.universe, [0, 0, 2, 4.5])
pressure['not_very_strong'] = fuzz.trimf(pressure.universe, [3, 5.5, 8])
pressure['strong'] = fuzz.trapmf(pressure.universe, [6.5, 8.5, 10, 10])

# 4. Функції належності для кутів повороту кранів [-90, 90]
for valve in [hot_valve, cold_valve]:
    valve['med_left'] = fuzz.trimf(valve.universe, [-65, -45, -25])
    valve['small_left'] = fuzz.trimf(valve.universe, [-35, -18, 0])
    valve['zero'] = fuzz.trimf(valve.universe, [-10, 0, 10])
    valve['small_right'] = fuzz.trimf(valve.universe, [0, 18, 35])
    valve['med_right'] = fuzz.trimf(valve.universe, [25, 45, 65])
    valve['large_right'] = fuzz.trapmf(valve.universe, [55, 75, 90, 90])

# Візуалізація функцій належності (графіки 1, 2, 3)
temp.view()
pressure.view()
hot_valve.view()

# 5. База з 11 правил (строго за текстом Завдання №1)
rules = [
    # 1. Гаряча і напір сильний -> гарячий середній вліво, холодний середній вправо
    ctrl.Rule(temp['hot'] & pressure['strong'], (hot_valve['med_left'], cold_valve['med_right'])),
    # 2. Гаряча і напір не дуже сильний -> холодний середній вправо
    ctrl.Rule(temp['hot'] & pressure['not_very_strong'], cold_valve['med_right']),
    # 3. Не дуже гаряча і напір сильний -> гарячий невеликий вліво
    ctrl.Rule(temp['not_very_hot'] & pressure['strong'], hot_valve['small_left']),
    # 4. Не дуже гаряча і напір слабий -> обидва невеликий вправо
    ctrl.Rule(temp['not_very_hot'] & pressure['weak'], (hot_valve['small_right'], cold_valve['small_right'])),
    # 5. Тепла і напір не дуже сильний -> залишити без змін
    ctrl.Rule(temp['warm'] & pressure['not_very_strong'], (hot_valve['zero'], cold_valve['zero'])),
    # 6. Прохолодна і напір сильний -> гарячий середній вправо, холодний середній вліво
    ctrl.Rule(temp['cool'] & pressure['strong'], (hot_valve['med_right'], cold_valve['med_left'])),
    # 7. Прохолодна і напір не дуже сильний -> гарячий середній вправо, холодний невеликий вліво
    ctrl.Rule(temp['cool'] & pressure['not_very_strong'], (hot_valve['med_right'], cold_valve['small_left'])),
    # 8. Холодна і напір слабий -> гарячий великий вправо
    ctrl.Rule(temp['cold'] & pressure['weak'], hot_valve['large_right']),
    # 9. Холодна і напір сильний -> гарячий середній вліво, холодний середній вправо
    ctrl.Rule(temp['cold'] & pressure['strong'], (hot_valve['med_left'], cold_valve['med_right'])),
    # 10. Тепла і напір сильний -> обидва невеликий вліво
    ctrl.Rule(temp['warm'] & pressure['strong'], (hot_valve['small_left'], cold_valve['small_left'])),
    # 11. Тепла і напір слабий -> обидва невеликий вправо
    ctrl.Rule(temp['warm'] & pressure['weak'], (hot_valve['small_right'], cold_valve['small_right'])),
]

# 6. Створення системи керування
water_ctrl = ctrl.ControlSystem(rules)
water_sim = ctrl.ControlSystemSimulation(water_ctrl)

# 7. Тестування на 4 різних сценаріях
test_cases = [
    (85, 9),    # Сценарій 1: Гаряча вода, сильний напір
    (45, 5.5),  # Сценарій 2: Тепла вода, помірний напір
    (15, 2),    # Сценарій 3: Холодна вода, слабкий напір
    (30, 6)     # Сценарій 4: Прохолодна вода, помірний напір
]

print("=== Тестування системи керування кранами ===")
for t_val, p_val in test_cases:
    water_sim.input['temp'] = t_val
    water_sim.input['pressure'] = p_val
    water_sim.compute()
    h_out = water_sim.output.get('hot_valve', 0.0)
    c_out = water_sim.output.get('cold_valve', 0.0)
    print(f"Темп: {t_val}°C, Напір: {p_val} -> Кран гарячої: {h_out:+.2f}°, Кран холодної: {c_out:+.2f}°")

# Візуалізація виводу для першого тестового випадку (85°C, напір 9)
water_sim.input['temp'] = 85
water_sim.input['pressure'] = 9
water_sim.compute()
hot_valve.view(sim=water_sim)
cold_valve.view(sim=water_sim)

plt.show()