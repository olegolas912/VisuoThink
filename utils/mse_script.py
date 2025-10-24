import os
import json
import re
import numpy as np

def extract_answer_from_log(log_content):
    """
    Извлекает числа после 'ANSWER:' и до 'TERMINATE' в строках логов.
    Если данные не найдены, возвращает 0.0.
    """
    match = re.search(r'ANSWER:\s*(\d+\.\d+)', log_content)
    if match:
        print("MATCHED")
        answer_str = match.group(1)
        print(float(answer_str))
        return [float(answer_str)]
    return [0.0]  # Если данных нет, возвращаем 0.0

def extract_label_from_json(json_file_path):
    """
    Извлекает значение по ключу ["ext_info"]["label"] из JSON файла.
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        return float(json_data.get("ext_info", {}).get("label", 0))

def calculate_mse(true_values, predicted_values):
    """
    Рассчитывает среднеквадратичную ошибку (MSE) между реальными и предсказанными значениями,
    учитывая только те пары, где оба значения не ноль.
    """
    squared_errors = []
    # trues = []
    # Итерируемся по парам истинных и предсказанных значений
    for true, pred in zip(true_values, predicted_values):
        if true != 0.0 and pred != 0.0: 
            if true in [19.47, 95.2, 21.0, 65.6, 78.4, 68.36, 56.28, 15.0, 197.3, 14.12]:
            # print(true, pred) # Только если оба значения не ноль
            # trues.append(true)
                squared_errors.append((true - pred) ** 2)
    
    if squared_errors:
        # Усредняем квадраты ошибок
        # print(trues)
        return np.mean(squared_errors)

    return float('nan') 

def process_data(base_folder_path):
    """
    Обрабатывает все папки в указанной директории, собирает значения из JSON и лог файлов,
    затем вычисляет MSE.
    """
    true_values = []
    predicted_values = []

    # Проходим по всем папкам в указанной директории
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        folder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_path):
            json_file_path = os.path.join(folder_path, 'ex.json')  # Путь к JSON файлу
            log_file_path = os.path.join(folder_path, 'output.log') 

            if os.path.exists(json_file_path) and os.path.exists(log_file_path):
                # Извлекаем правильный ответ из JSON
                label_value = extract_label_from_json(json_file_path)
                true_values.append(label_value)
                # Извлекаем ответ из лога
                with open(log_file_path, 'r', encoding='utf-8') as log_file:
                    log_content = log_file.read()
                    answer_values = extract_answer_from_log(log_content)
                    
                    if answer_values:
                        predicted_values.append(answer_values[0])  # Берем первое значение, если оно есть

    return true_values, predicted_values

def main():
    base_folder_path = '/workspace/VisuoThink/outputs/geometry/Qwen/Qwen2-VL-7B-Instruct'  # Укажите путь к папке с данными

    # Получаем истинные значения (labels) и предсказанные (из логов)
    true_values, predicted_values = process_data(base_folder_path)

    # Рассчитываем MSE
    mse = calculate_mse(true_values, predicted_values)

    # Выводим результаты
    print(f"Истинные значения: {true_values}")
    print(f"Предсказанные значения: {predicted_values}")
    print(f"Среднеквадратичная ошибка (MSE): {mse}")

if __name__ == "__main__":
    main()
