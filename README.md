![logo](https://github.com/Druidblack/yandex_music_nowplaying/blob/main/img/unnamed%20копия.png?raw=true)
## Yandex Music Now Playing

![2](https://github.com/Druidblack/yandex_music_nowplaying/blob/main/img/2.jpg)
![1](https://github.com/Druidblack/yandex_music_nowplaying/blob/main/img/1.jpg)

Получаем данные о воспроизводимом треке с вашего Яндекс-аккаунта на любом устройстве. Получаем переключатель которым можно лайкать треки (или снимать лайк). Также мы автоматически записываем воспроизводимые треки с любых устройств на Last.fm.

Опрос кнопки лайка 60 секунд. Если нажмете, лайк уйдет сразу. 60 секунд нужно для опроса уже лайкнутой песни. Если нажмете раньше опроса лайк на уже лайкнутую песню, то ничего не произойдет, переключатель перейдет в включенный режим (лайн не удалится).

Отправка данных на last.fm опциональная. Сдела альтернативу (после того как яндекс убрал отправку данных на last.fm) для отправки данных с колонок.

Получение даннных с яндекс колонок работает с интеграцией https://github.com/AlexxIT/YandexStation Из нее берем колонки.

```yaml
sensor:
  - platform: yandex_music_nowplaying
    name: Yandex Music Now Playing
    token: AgAAAAACO3_rAAG8X79EtXkI30lLpjij1FFUQ
    scan_interval: '00:00:10'
    ynison: true
    station_entities: "media_player.yandex_station_l015a4b0016grv, media_player.yandex_station_l81a1s70061asz"
    
    lastfm:
      enabled: true
      api_key: f7939d0ae9ff6700f5
      api_secret: 057d7d9e7f2ba67710d
      session_key: A--WWj1WxCQLNT6D5
      # min_scrobble_percent: 50
      # min_scrobble_seconds: 240
    
switch:
  - platform: yandex_music_nowplaying
    name: Yandex Music Like
```
Создаем приложение https://www.last.fm/api/account/create полученные данные нужны для сенсора и для получения session_key

session_key можно получить по ссылке https://dullmace.github.io/lastfm-sessionkey/
