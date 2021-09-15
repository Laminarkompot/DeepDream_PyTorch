# DeepDream_PyTorch
Простая и понятная реализация Deep Dream в PyTorch

**Матчасть**

Как известно, Deep Dream представляет собой задачу "обучения" изображения, подаваемого на вход обученной свёрточной нейросети с целью максимизации активаций её фильтров (функции потерь) и получения на этом изображении забавных психоделических эффектов. Конкретно здесь задействована сеть `ResNet18` из пакета `totchvision.models`, а в качестве функции потерь используется взвешенная сумма L2 норм срезов активаций её модулей `layer3` и `layer4`.

**Как пользоваться?**

Открываем файл, меняем каталоги и названия файлов в начале на желаемые и запускаем скрипт как `main`. 
```
# Overwrite these values before launching script
PATH_TO_FILE = '/home/username/Downloads/picture.jpg'
SAVE_PATH = '/home/username/Downloads/picture_dream.png'
```
Также можно поэкспериментировать с этими параметрами, чтобы получаемая картинка выглядела наиболее интересным образом:
```
# Deep Dream adjustable parameters (try to toggle for better result)
ORIG_IMAGE_INTENSITY = 0.8
DREAM_INTENSITY = 0.09
RED_CH_FACTOR = 0.4
GREEN_CH_FACTOR = 0.3
BLUE_CH_FACTOR = 0.2
LAYER_3_CONTRIBUTION = 2
LAYER_4_CONTRIBUTION = 20

# Result picture dimensions (400x400 recommended)
H = 400
W = 400
```
