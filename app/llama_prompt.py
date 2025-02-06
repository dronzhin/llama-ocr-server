import ollama
from pydantic import BaseModel
import json
import pandas as pd

class BloodTest(BaseModel):
  indicator: str
  result: str
  measurement: str | None

class BloodTestList(BaseModel):
  indicators: list[BloodTest]

# Запрос к модели Llama 3.2 3B
def llama3b(message, temperature = 0.0):
    response = ollama.generate(
        model="llama3.2:3b",
        prompt=message,  # Передаём сообщение как prompt
        options={
            "temperature": temperature  # Передаём temperature в options
        },
        format=BloodTestList.model_json_schema()
    )
    # Преобразование в json
    json_string = response.response
    result = json.loads(json_string)
    result = result['indicators']

    # Преобразование в Pandas
    result = pd.DataFrame(result)

    # Заменяем названия столбцов
    new_columns = ["Название", "Значение", "Ед. Измерения"]
    result.columns = new_columns

    return result


def llama11b(message, path_image, temperature = 0.0):
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': message,
            'images': [path_image],
            "temperature": temperature
        }],
        format=BloodTestList.model_json_schema()
    )

    # Преобразование в json
    json_string = response.message.content
    result = json.loads(json_string)
    result = result['indicators']

    # Преобразование в Pandas
    result = pd.DataFrame(result)

    # Заменяем названия столбцов
    new_columns = ["Название", "Значение", "Ед. Измерения"]
    result.columns = new_columns

    return result


if __name__ == "__main__":
    text = 'Гамоглобин составляет 3 единицы'
    res = llama3b(text)
    print(res)