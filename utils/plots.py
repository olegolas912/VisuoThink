# import seaborn as sns
# import matplotlib.pyplot as plt

# # Два словаря с MSE значениями
# mse_set_1 = {
#     "qwen2": 5478.1431625000005,
#     "qwen3": 2089.1181162162165,
#     "internvl3_5": 4448.246654545455,
# }

# mse_set_2 = {
#     "qwen3": 385.7784636363635,
#     "internvl3_5": 421.65339999999986,
# }

# # Преобразуем словари в списки для использования с Seaborn
# data_set_1 = list(mse_set_1.items())
# data_set_2 = list(mse_set_2.items())

# # Создание DataFrame для Seaborn
# import pandas as pd

# df_set_1 = pd.DataFrame(data_set_1, columns=['Metric', 'MSE'])
# df_set_2 = pd.DataFrame(data_set_2, columns=['Metric', 'MSE'])

# # Построение первого графика с Seaborn
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Metric', y='MSE', data=df_set_1, palette='Blues_d')
# plt.title('MSE for Set 1')
# plt.xlabel('Models')
# plt.ylabel('MSE')
# plt.savefig('mse_set_1_seaborn.png')  # Сохраняем график для набора 1
# plt.show()

# # Построение второго графика с Seaborn
# plt.figure(figsize=(8, 6))
# sns.barplot(x='Metric', y='MSE', data=df_set_2, palette='Greens_d')
# plt.title('MSE for Set 2')
# plt.xlabel('Models')
# plt.ylabel('MSE')
# plt.savefig('mse_set_2_seaborn.png')  # Сохраняем график для набора 2
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Данные
true_values_qwen2 = [19.47, 23.34, 186.53, 15.66, 95.2, 10.29, 22.51, 21.0, 24.95, 99.56, 65.6, 16.4, 181.56, 78.4, 79.2, 20.26, 16.45, 13.71, 68.36, 22.05, 15.0, 125.5, 12.01, 56.28, 43.2, 24.0, 15.0, 108.0, 75.4, 197.3, 14.12, 131.28, 42.88]
predicted_values_qwen2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

true_values_qwen3 = [19.47, 23.34, 186.53, 15.66, 95.2, 10.29, 22.51, 21.0, 24.95, 99.56, 65.6, 16.4, 181.56, 78.4, 79.2, 20.26, 16.45, 13.71, 68.36, 22.05, 15.0, 125.5, 12.01, 56.28, 43.2, 24.0, 15.0, 108.0, 75.4, 197.3, 14.12, 131.28, 42.88]
predicted_values_qwen3 = [19.54, 21.92, 0.0, 0.0, 96.86, 0.0, 22.5, 21.0, 14.06, 24.0, 65.69, 16.63, 200.87, 88.0, 47.49, 6.89, 16.28, 9.0, 68.36, 0.0, 15.0, 0.0, 12.28, 52.87, 44.22, 24.0, 15.0, 109.31, 23.84, 261.5, 10.2, 196.0, 0.0]

true_values_internvl = [19.47, 23.34, 186.53, 15.66, 95.2, 10.29, 22.51, 21.0, 24.95, 99.56, 65.6, 16.4, 181.56, 78.4, 79.2, 20.26, 16.45, 13.71, 68.36, 22.05, 15.0, 125.5, 12.01, 56.28, 43.2, 24.0, 15.0, 108.0, 75.4, 197.3, 14.12, 131.28, 42.88]
predicted_values_internvl = [19.54, 0.0, 0.0, 0.0, 96.85, 0.0, 0.0, 21.0, 0.0, 0.0, 65.68, 0.0, 0.0, 88.0, 0.0, 0.0, 0.0, 0.0, 68.36, 0.0, 0.0, 0.0, 0.0, 56.29, 0.0, 0.0, 15.0, 0.0, 0.0, 261.5, 14.13, 0.0, 0.0]

# Приведение всех списков к одинаковой длине, заполняем нулями там, где данные отсутствуют
max_len = max(len(true_values_qwen2), len(predicted_values_qwen2), len(true_values_qwen3), len(predicted_values_qwen3), len(true_values_internvl), len(predicted_values_internvl))

# Функция для выравнивания длины данных
def align_length(true_values, predicted_values, max_len):
    true_values += [0.0] * (max_len - len(true_values))
    predicted_values += [0.0] * (max_len - len(predicted_values))
    return true_values, predicted_values

# Приведение всех данных к одинаковой длине
true_values_qwen2, predicted_values_qwen2 = align_length(true_values_qwen2, predicted_values_qwen2, max_len)
true_values_qwen3, predicted_values_qwen3 = align_length(true_values_qwen3, predicted_values_qwen3, max_len)
true_values_internvl, predicted_values_internvl = align_length(true_values_internvl, predicted_values_internvl, max_len)

# Создание DataFrame для Seaborn
data_qwen2 = pd.DataFrame({
    'Index': range(len(true_values_qwen2)),
    'True Values': true_values_qwen2,
    'Predicted Values': predicted_values_qwen2
})

data_qwen3 = pd.DataFrame({
    'Index': range(len(true_values_qwen3)),
    'True Values': true_values_qwen3,
    'Predicted Values': predicted_values_qwen3
})

data_internvl = pd.DataFrame({
    'Index': range(len(true_values_internvl)),
    'True Values': true_values_internvl,
    'Predicted Values': predicted_values_internvl
})

# Построение графика
plt.figure(figsize=(10, 6))

# Построение линий для Qwen2
sns.lineplot(x='Index', y='True Values', data=data_qwen2, label='True Values Qwen2', linewidth=3, color='#4682B4')  # SteelBlue
sns.lineplot(x='Index', y='Predicted Values', data=data_qwen2, label='Predicted Qwen2', linestyle='--', color='#4682B4')

# Построение линий для Qwen3
sns.lineplot(x='Index', y='True Values', data=data_qwen3, label='True Values Qwen3', linewidth=3, color='#B0C4DE')  # MediumSlateBlue
sns.lineplot(x='Index', y='Predicted Values', data=data_qwen3, label='Predicted Qwen3', linestyle='--', color='#B0C4DE')

# Построение линий для Internvl
sns.lineplot(x='Index', y='True Values', data=data_internvl, label='True Values Internvl', linewidth=3, color='#7B68EE')  # MediumSlateBlue
sns.lineplot(x='Index', y='Predicted Values', data=data_internvl, label='Predicted Internvl', linestyle='--', color='#7B68EE')

# Заголовок и подписи осей
plt.title('Comparison of True and Predicted Values for Different Models', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Values', fontsize=12)

# Легенда
plt.legend()

# Сохранение графика
plt.savefig('comparison_true_predicted_values_seaborn_updated.png')

# Показать график
plt.show()
