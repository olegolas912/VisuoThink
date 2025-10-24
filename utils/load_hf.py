from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import get_peft_model, LoraConfig
import os
from PIL import Image

# Шаг 1: Загрузка основной модели (backbone)
backbone_model_name = "Qwen/Qwen3-VL-4B-Instruct"  # Основная модель
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-Instruct")
model = AutoModelForVision2Seq.from_pretrained(backbone_model_name)

# Шаг 2: Загрузка модели LoRA
lora_checkpoint_path = "/workspace/qwen3vl-geoqa-lora/checkpoint-30"  # Путь к LoRA адаптеру
lora_cfg = LoraConfig.from_pretrained(lora_checkpoint_path)

# Шаг 3: Применение LoRA адаптера к основной модели
model_with_lora = get_peft_model(model, lora_cfg)

# Слияние LoRA адаптера с основной моделью
final_model = model_with_lora.merge_and_unload()  # Эта функция сливает LoRA и освобождает адаптер

# # Шаг 4: Сохранение итоговой модели в папку
output_dir = "./combined_model"
processor.save_pretrained(output_dir)
final_model.save_pretrained(output_dir)

# Шаг 5: Загрузка на Hugging Face
# Логинимся через командную строку перед загрузкой
# transformers-cli login

# Затем загружаем модель
from huggingface_hub import upload_folder

# Убедитесь, что на Hugging Face у вас есть репозиторий, куда можно загрузить модель.
repo_name = "Kate-03/Qwen3-VL-4B-Geo170k"
upload_folder(
    repo_id=repo_name,
    folder_path=output_dir,
    path_in_repo="."  # Папка, куда будет загружена модель в репозитории
)

print(f"✅ Модель загружена на Hugging Face: {repo_name}")
