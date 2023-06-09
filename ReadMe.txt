ПРОГНОЗИРОВАНИЕ СОСТОЯНИЯ ЦЕНЫ ПО ВЫБРАННОЙ КРИПТОВАЛЮТНОЙ ПАРЕ

* Источник данных котировок: binance.com;
* Программа выполнена в виде проекта PyCharm IDE;
* Для примера выполнения программы выбрана торговая пара LTC/USDT и временной интервал в 15 мин.;
* В качестве примера программа выполняет прогноз 11-й "свечи" соответствующего временного интервала на основе данных 10-ти предыдущих;
* Все ключевые моменты алгоритма прокомментированы в коде.

Порядок работы:
1.  По усмотрению пользователя, происходит обновление (создание) базы котировок выбранной для примера работы программы торговой пары (LTC/USDT).
    Информация получается при помощи Binance API и библиотеки Requests.
    Полученные данные формируют NumPy-массив.
2.  По усмотрению пользователя, происходит создание или обновление (с сохранением в спец. файл с расширением .nn) ИНС (иск. нейронной сети).
    Для ее обучения используются данные из записанного ранее .csv-файла, которые подразделяются на массивы входных данных и данных для проверки.
    Также в процессе обучения выборка подразделяется на тренировочную чатсь и тестовую.
    После процесса обучения выводятся графики выбранных метрик обучения.
    Для наглядности на графике каждой метрики представлена совмещенная динамика обучения модели на тренировочной части выборки и тестовой.
3.  По усмотрению пользователя, программа может продемонстрировать прогнозирование динамики цены на оперделенном временном интервале.
    Прогнозирование совершается для последнего часа полученных данных (4 свечи по 15 минут - выбранный пример работы программы).
    Каждая последующая свеча формируется все больше на основе уже спрогнозированных созданной моделью данных.
    Для наглядности, совместно со "свечным" графиком оригинальной динамики цены, представлена линия динамики цены закрытия, спрогнозированная программой.
