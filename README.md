# mlops_LisinFedor

# Установка пакета

Для работы с проектом как с программой после скачивания репозитория выполнить из корня (желательно в окружении):
```
pip install .
```

Для изменения проекта стоит установить (синтаксис может быть немного другой, если использовать zsh, а не bash)

```
pip install -e .[dev,tests]
```

# .env и DVC
Для корректной работы следует создать `.env` файл и установить следующие переменные среды (за доступом при приверке ДЗ обращаться к владельцу репозитория):
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACESS_KEY
- ENDPOINT_URL
- MLFLOW_URI

Первые три переменных необязательны, если настроен DVC (см. ниже).

Так же следует создать `.dvc/config.local` и настроить переменные в файле аналогично `.dvc/config` (за доступом при приверке ДЗ обращаться к владельцу репозитория).


# Classification

Вызвать обучение или предсказания можно находясь в любое месте с активированным окружением (в котором установлен пакет) через команду 
`classification`. 

```
usage: classification [-h] (--train | --predict) [--modelid MODELID | --modelpath MODELPATH] [--outfile OUTFILE] [--input INPUT]
                 [--modelname MODELNAME] [--mlflow | --no-mlflow]

optional arguments:
  -h, --help            show this help message and exit
  --train               run train pipline using config.yml file.
  --predict             make prediction on --input using --modelid or last (from config.yml) saved model; store to --outfile
  --modelid MODELID     id of model from mlflow storage
  --modelpath MODELPATH
                        path to model; specify only one of the two arguments 'id' or 'path'
  --outfile OUTFILE     file path to store prediction in 'my/path/file.csv' format; required argument for prediction
  --input INPUT         path to data file for prediction; required argument for prediction
  --modelname MODELNAME
                        this name will be used for saving model, kwargs config and mlflow; default config.model.model_name
  --mlflow              store model and artifacts into mlflow; if not provided use_mlflow parameter from config will be used
  --no-mlflow           prevent saving experiment to mlflow; if not provided use_mlflow parameter from config will be used
```

## Train
Команда `classification --train [--mlflow | --no-mlflow] [--modelname MODELNAME]`.
Для тренировки используются данные из параметра `data_path` конфигурационного файла.

### Если настроены параметры AWS в .env

Если параметр не задан, будет использован путь до данных, которые скачиваются из S3 `${s3.defaultout} / ${s3.defaultfile}`. Если 
данных нет, они будут скачаны из S3 автоматически.

### Если настроен DVC конфиг

При отсутсвии данных выполнить `dvc pull`.

### Конфигурация модели

Для использования той или иной модели в обучении следует настроить следующие параметры:

- module: название модуля/пакета где лежит есть класс модели (прим.: sklearn)
- submodule: модуль пакета (прим.: ensemble)
- model_name: класс модели (прим.: GradientBoostingClassifier)

Задать конфигурацию модели перед обучением можно добавлением файла конфигурации в `/configs/model_configs`
в формате `*_kwargs.yml`. Имя файла должно начинаться с названия модели, указанного в файле `/configs/config.yml`. Название модели параметр `model.save_as`, если он не задан используется `model.model_name`. Если при обучении файл конфигурации не найден, он будет создан автоматически со стандартной конфигурацией, после чего его можно будет изменить и новое обучение будет использовать изменнный файл.

### Трансформеры

В проекте используются 2 кастомных трансформера:
- CatTransformer: 

    трансформирует категориальные фичи в dummy вектора. Тип фичи (категориальная или нет) определяется автоматически по числу уникальных значений. Граница определяется параметром categorical_features_max_uniqs

- ScalerTransformer: 
    
    делает скейлинг numeric фичей. Тип фичи определяется автоматически по числу уникальных значений и питоновскому типу, граница задатся через numeric_features_min_uniqs. Для нормализации используются стандартные скейлеры из sklearn, задатся через параметр scaler


### mlflow

Если задан URL то модели можно сохранять в mlflow. Стандартные модели `sklearn` сохраняться как модель, нестандартные - как артифакт. При сохранении модели в mlflow путь до артифакта будет записан в конфиг как `model.last_model`.

Если при запуске скрипта будет передн один из аргументов `--mlflow` или `--no-mlflow`, то параметр конфигурации `use_mlfow` будет проигнорирован.

### dvc repro

Если настроен DVC то для получения обученной модели можно выполнить `dvc repro`. Будет обучена модель `last`, с конфигурацией `last_kwargs.yml`.

## Prediction
Команда `classification --predict [--modelid MODELID | --modelpath MODELPATH] (--outfile OUTFILE) (--input INPUT)`.

Если `modelid` или `modelpath` не задан, по умолчанию будет использована последняя загруженная в mlflow модель (путь в mlflow берётся из конфигурации).

# Final

0. Описание в README.md (+ 1 балл)
1. Это и есть оценка. Будет скопировано в PR (+ 1 балл)
2. Ноутбук с EDA есть (+ 1 балл)
3. Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (+ 3 балла)
4. Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (+ 3 балла)
5. Проект имеет модульную структуру (+ 2 балла)
6. Использованы логгеры (+ 2 балла)
7. Написаны тесты на отдельные модули и на прогон обучения и predict (+ 3 балла)
8. Для тестов генерируются синтетические данные, приближенные к реальным (+ 2 балла)
9. Обучение модели конфигурируется с помощью конфигов в json или yaml, добавил две конфигурации для разных классов моделей (+ 3 балла)
10. Используются датаклассы для сущностей из конфига, а не голые dict (+ 2 балла)
11. Напишите кастомный трансформер (есть целых два) и протестируйте его (+ 3 балла) 
12. В проекте зафиксированы все зависимости (в setup.py) (+ 1 балл) 
13. Настроен CI для прогона тестов, линтера на основе github actions (+ 3 балла).
14. Используйте hydra для конфигурирования - не вышло настроить гидру для проекта, в котором при запуске происходит парсинг аргументов, т.е. неполучиться использовать @hydra.main(). Было бы интересно посмотреть как действовать в такм случае (например через compose и initialize) - + 0 баллов
15. разверните локально mlflow или на какой-нибудь виртуалке (+ 1 балл)
16. залогируйте метрики (+ 1 балл)
17. воспользуйтесь Model Registry для регистрации модели. Скриншота не будет, просто дам ссылку на mlflow с моделями (+ 1 балл)
18. выделите в своем проекте несколько entrypoints в виде консольных утилит (+ 1 балл).
19. добавьте датасет под контроль версий (+ 1 балл)
20. сделайте dvc пайплайн для изготовления модели(+ 1 балл)
