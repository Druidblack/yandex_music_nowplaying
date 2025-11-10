# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import random
import re
import string
import time
from datetime import timedelta
from typing import Any, Callable, Optional

import voluptuous as vol

from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import (
    CONF_NAME,
    CONF_PASSWORD,
    CONF_SCAN_INTERVAL,
    CONF_TOKEN,
    CONF_USERNAME,
)
from homeassistant.core import HomeAssistant, ServiceCall, State, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change
from homeassistant.helpers.typing import DiscoveryInfoType
from homeassistant.helpers.aiohttp_client import async_get_clientsession

import hashlib

_LOGGER = logging.getLogger(__name__)  # custom_components.yandex_music_nowplaying.sensor

DEFAULT_NAME = "Yandex Music Now Playing"
DOMAIN = "yandex_music_nowplaying"

# -------- Last.fm (опционально) --------
LASTFM_SCHEMA = vol.Schema(
    {
        vol.Optional("enabled", default=True): cv.boolean,
        vol.Required("api_key"): cv.string,
        vol.Required("api_secret"): cv.string,
        vol.Required("session_key"): cv.string,
        vol.Optional("min_scrobble_percent", default=50): vol.All(int, vol.Range(min=1, max=100)),
        vol.Optional("min_scrobble_seconds", default=240): vol.All(int, vol.Range(min=30, max=3600)),
    }
)

# -------- Конфиг --------
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_TOKEN): cv.string,  # token ИЛИ username+password
        vol.Optional(CONF_USERNAME): cv.string,
        vol.Optional(CONF_PASSWORD): cv.string,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_SCAN_INTERVAL, default=timedelta(seconds=15)): cv.time_period,
        vol.Optional("update_interval"): vol.All(int, vol.Range(min=5, max=3600)),
        vol.Optional("ynison", default=True): cv.boolean,  # push с Я.Музыки (браузер/телефон)
        vol.Optional("ynison_mode", default="put"): vol.In(["auto", "get", "put"]),  # совместимость; фактически 'put'
        # Список колонок AlexxIT/YandexStation, из которых читать now playing
        vol.Optional("station_entities"): vol.Any(cv.string, [cv.string]),
        # Опционально: Last.fm
        vol.Optional("lastfm"): LASTFM_SCHEMA,
    }
)

# ---- кэш лайков (общий для sensor/switch) ----
LIKES_TTL = 60  # сек


def _likes_cache_get(hass: HomeAssistant):
    data = hass.data.setdefault(DOMAIN, {})
    return data.get("liked_cache"), data.get("liked_cache_ts", 0.0)


def _likes_cache_set(hass: HomeAssistant, liked_set: set[str]):
    data = hass.data.setdefault(DOMAIN, {})
    data["liked_cache"] = liked_set
    data["liked_cache_ts"] = time.time()


def _likes_cache_touch_add(hass: HomeAssistant, track_id: Any):
    if not track_id:
        return
    s, _ = _likes_cache_get(hass)
    if s is None:
        s = set()
    s.add(str(track_id))
    _likes_cache_set(hass, s)


def _likes_cache_touch_remove(hass: HomeAssistant, track_id: Any):
    if not track_id:
        return
    s, _ = _likes_cache_get(hass)
    if s is None:
        s = set()
    s.discard(str(track_id))
    _likes_cache_set(hass, s)


# ---------------- Last.fm Scrobbler (опционально) ----------------
class LastFmScrobbler:
    """
    Лёгкий клиент Last.fm:
      - updateNowPlaying при смене трека
      - планирование scrobble по порогу min(%, сек) от старта
    Только aiohttp-сессия HA; без блокирующих вызовов.
    """

    API_URL = "https://ws.audioscrobbler.com/2.0/"

    def __init__(
        self,
        hass: HomeAssistant,
        api_key: str,
        api_secret: str,
        session_key: str,
        min_scrobble_percent: int = 50,
        min_scrobble_seconds: int = 240,
    ) -> None:
        self.hass = hass
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_key = session_key
        self.min_percent = max(1, min(100, int(min_scrobble_percent)))
        self.min_seconds = max(30, min(3600, int(min_scrobble_seconds)))

        self._session = async_get_clientsession(hass)
        self._current_key: Optional[str] = None
        self._last_nowplaying_key: Optional[str] = None
        self._last_scrobbled_key: Optional[str] = None
        self._scrobble_task: Optional[asyncio.Task] = None

    @staticmethod
    def _build_track_key(artist: str, title: str, album: Optional[str], duration_sec: Optional[float]) -> str:
        a = (artist or "").strip()
        t = (title or "").strip()
        al = (album or "").strip() if album else ""
        d = int(duration_sec) if duration_sec else 0
        return f"{a}\n{t}\n{al}\n{d}"

    @staticmethod
    def _as_int_seconds(x: Optional[float | int]) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(round(float(x)))
        except Exception:
            return None

    def _sign(self, params: dict) -> str:
        # Сигнатура: конкат всех key+value в алф. порядке ключей + secret, md5 hex
        concat = "".join(k + str(params[k]) for k in sorted(params.keys()))
        concat += self.api_secret
        return hashlib.md5(concat.encode("utf-8")).hexdigest()

    async def _post(self, params: dict) -> dict | None:
        # Готовим подпись (без format)
        params = dict(params)
        params["api_key"] = self.api_key
        params["sk"] = self.session_key
        params["format"] = "json"
        sign_params = {k: v for k, v in params.items() if k not in ("format", "api_sig")}
        params["api_sig"] = self._sign(sign_params)

        try:
            async with self._session.post(self.API_URL, data=params, timeout=20) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    _LOGGER.debug("Last.fm HTTP %s: %s", resp.status, txt[:300])
                    return None
                try:
                    return json.loads(txt)
                except Exception:
                    _LOGGER.debug("Last.fm JSON parse failed: %s", txt[:200])
                    return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _LOGGER.debug("Last.fm post error: %r", e)
            return None

    async def _update_now_playing(self, artist: str, title: str, album: Optional[str], duration_sec: Optional[int]) -> None:
        params = {
            "method": "track.updateNowPlaying",
            "artist": artist or "",
            "track": title or "",
        }
        if album:
            params["album"] = album
        if duration_sec is not None and duration_sec > 0:
            params["duration"] = str(duration_sec)
        await self._post(params)

    async def _scrobble(self, artist: str, title: str, album: Optional[str], timestamp: int, duration_sec: Optional[int]) -> None:
        # track.scrobble использует массивы param[0]
        params = {
            "method": "track.scrobble",
            "artist[0]": artist or "",
            "track[0]": title or "",
            "timestamp[0]": str(int(timestamp)),
        }
        if album:
            params["album[0]"] = album
        if duration_sec is not None and duration_sec > 0:
            params["duration[0]"] = str(duration_sec)
        await self._post(params)

    def _cancel_pending_scrobble(self) -> None:
        if self._scrobble_task and not self._scrobble_task.done():
            self._scrobble_task.cancel()
        self._scrobble_task = None

    def shutdown(self) -> None:
        self._cancel_pending_scrobble()

    def on_track_update(
        self,
        *,
        artist: Optional[str],
        title: Optional[str],
        album: Optional[str],
        duration_ms: Optional[int],
        progress_sec: Optional[float] = None,
    ) -> None:
        """Вызывается сенсором при каждом апдейте playing."""
        artist_s = (artist or "").strip()
        title_s = (title or "").strip()
        album_s = (album or "").strip() if album else None
        duration_sec = self._as_int_seconds((duration_ms or 0) / 1000.0 if duration_ms else None)

        if not artist_s and not title_s:
            return

        key = self._build_track_key(artist_s, title_s, album_s, duration_sec)
        if key != self._current_key:
            # Новый трек: отменяем прошлый таймер и отправляем Now Playing
            self._current_key = key
            self._cancel_pending_scrobble()

            now = time.time()
            prog = float(progress_sec) if progress_sec is not None else 0.0
            start_ts = now - max(0.0, prog)

            if duration_sec and duration_sec > 0:
                threshold = min(self.min_seconds, int(duration_sec * (self.min_percent / 100.0)))
            else:
                threshold = self.min_seconds

            fire_ts = int(start_ts + threshold)
            delay = max(0.0, fire_ts - now)

            # Обновляем Now Playing (только если ещё не отправляли для этого ключа)
            if key != self._last_nowplaying_key:
                self._last_nowplaying_key = key
                self.hass.loop.create_task(self._update_now_playing(artist_s, title_s, album_s, duration_sec))

            # Если ещё не скроббили этот ключ — планируем отправку
            if key != self._last_scrobbled_key:
                async def _scrobble_later():
                    try:
                        await asyncio.sleep(delay)
                        if self._current_key != key:
                            return
                        await self._scrobble(artist_s, title_s, album_s, fire_ts, duration_sec)
                        self._last_scrobbled_key = key
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        _LOGGER.debug("Last.fm scrobble error: %r", e)

                self._scrobble_task = self.hass.loop.create_task(_scrobble_later())
        else:
            pass


# -------- Ynison watcher (WS, Put-only — для браузера/телефона) --------
class YnisonWatcher:
    """
    Подписка на Ynison через PutYnisonState (теневое устройство).
    Это стабильный вариант для браузера/телефона.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        token: str,
        yaclient: Any,
        on_update: Callable[[dict | None], None],
    ):
        self.hass = hass
        self._token = token
        self._client = yaclient
        self._on_update = on_update
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopped.clear()
        self._task = self.hass.loop.create_task(self._runner())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3)
            except asyncio.TimeoutError:
                pass

    async def _runner(self) -> None:
        backoff = 2
        while not self._stopped.is_set():
            try:
                ok = await self._connect_once()
                if ok:
                    backoff = 2
                else:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
            except asyncio.CancelledError:
                return
            except Exception as e:
                _LOGGER.debug("Ynison loop error: %r", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _connect_once(self) -> bool:
        import aiohttp

        device_id = "".join(random.choice(string.ascii_lowercase) for _ in range(16))
        device_info = {"app_name": "HomeAssistant", "type": 1}
        ws_proto = {
            "Ynison-Device-Id": device_id,
            "Ynison-Device-Info": json.dumps(device_info),
        }

        headers_redirect = {
            "Sec-WebSocket-Protocol": f"Bearer, v2, {json.dumps(ws_proto)}",
            "Origin": "http://music.yandex.ru",
            "Authorization": f"OAuth {self._token}",
        }

        timeout = aiohttp.ClientTimeout(total=None, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # redirect → host + ticket
            try:
                async with session.ws_connect(
                    url="wss://ynison.music.yandex.ru/redirector.YnisonRedirectService/GetRedirectToYnison",
                    headers=headers_redirect,
                    timeout=10,
                ) as ws_redirect:
                    try:
                        recv = await ws_redirect.receive(timeout=10)
                    except asyncio.TimeoutError:
                        _LOGGER.debug("Ynison redirect receive timeout")
                        return False
                    data = json.loads(recv.data)
                    host = data.get("host")
                    ticket = data.get("redirect_ticket")
                    if not host or not ticket:
                        _LOGGER.debug("Ynison redirect failed: %s", data)
                        return False
            except asyncio.TimeoutError:
                _LOGGER.debug("Ynison redirect connect timeout")
                return False

            new_ws_proto = dict(ws_proto)
            new_ws_proto["Ynison-Redirect-Ticket"] = ticket

            headers = {
                "Sec-WebSocket-Protocol": f"Bearer, v2, {json.dumps(new_ws_proto)}",
                "Origin": "http://music.yandex.ru",
                "Authorization": f"OAuth {self._token}",
            }

            url = f"wss://{host}/ynison_state.YnisonStateService/PutYnisonState"
            async with session.ws_connect(url=url, headers=headers, timeout=10) as ws:
                _LOGGER.debug("Ynison connected (Put): host=%s device_id=%s", host, device_id)

                bootstrap = {
                    "update_full_state": {
                        "player_state": {
                            "player_queue": {
                                "current_playable_index": -1,
                                "entity_id": "",
                                "entity_type": "VARIOUS",
                                "playable_list": [],
                                "options": {"repeat_mode": "NONE"},
                                "entity_context": "BASED_ON_ENTITY_BY_DEFAULT",
                                "version": {
                                    "device_id": device_id,
                                    "version": 0,
                                    "timestamp_ms": 0,
                                },
                                "from_optional": "",
                            },
                            "status": {
                                "duration_ms": 0,
                                "paused": True,
                                "playback_speed": 1,
                                "progress_ms": 0,
                                "version": {
                                    "device_id": device_id,
                                    "version": 0,
                                    "timestamp_ms": 0,
                                },
                            },
                        },
                        "device": {
                            "capabilities": {
                                "can_be_player": True,
                                "can_be_remote_controller": False,
                                "volume_granularity": 16,
                            },
                            "info": {
                                "device_id": device_id,
                                "type": "WEB",
                                "title": "Home Assistant",
                                "app_name": "HomeAssistant",
                            },
                            "volume_info": {"volume": 0},
                            "is_shadow": True,
                        },
                        "is_currently_active": False,
                    },
                    "rid": device_id,
                    "player_action_timestamp_ms": 0,
                    "activity_interception_type": "DO_NOT_INTERCEPT_BY_DEFAULT",
                }
                await ws.send_str(json.dumps(bootstrap))

                async def _ping_loop():
                    try:
                        while True:
                            await asyncio.sleep(25)
                            await ws.ping()
                    except Exception:
                        pass

                ping_task = asyncio.create_task(_ping_loop())

                try:
                    while not self._stopped.is_set():
                        try:
                            msg = await ws.receive(timeout=60)
                        except asyncio.TimeoutError:
                            continue

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                await self._handle_state(json.loads(msg.data))
                            except Exception as e:
                                _LOGGER.debug("Ynison parse error: %r", e)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            return False
                        else:
                            continue
                finally:
                    ping_task.cancel()
            return True

    async def _handle_state(self, data: dict) -> None:
        ps = data.get("player_state") or {}
        q = ps.get("player_queue") or {}
        idx = q.get("current_playable_index", -1)
        lst = q.get("playable_list") or []
        playable_id = None

        if isinstance(idx, int) and 0 <= idx < len(lst):
            playable = lst[idx] or {}
            playable_id = playable.get("playable_id") or playable.get("id")

        track_id = self._normalize_track_id(playable_id)
        if not track_id:
            self._on_update(None)
            return

        tr_list = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._client.tracks(track_id)
        )
        track = tr_list[0] if isinstance(tr_list, list) and tr_list else None
        if not track:
            self._on_update(None)
            return

        artists = (
            ", ".join([a.name for a in getattr(track, "artists", [])])
            if getattr(track, "artists", None)
            else None
        )
        album_title = (
            getattr(track.albums[0], "title", None)
            if getattr(track, "albums", None)
            else None
        )

        cover = None
        cover_uri = getattr(track, "cover_uri", None)
        if cover_uri:
            uri = cover_uri.replace("%%", "300x300")
            cover = uri if uri.startswith("http") else f"https://{uri}"

        update = {
            "title": getattr(track, "title", None),
            "artists": artists,
            "album": album_title,
            "track_id": getattr(track, "id", None),
            "cover": cover,
            "duration_ms": getattr(track, "duration_ms", None),
            "explicit": getattr(track, "explicit", None),
            "context_type": q.get("entity_type"),
            "queue_id": q.get("entity_id"),
        }
        self._on_update(update)

    @staticmethod
    def _normalize_track_id(playable_id: Any) -> Optional[str]:
        if not playable_id:
            return None
        s = str(playable_id)
        if s.isdigit():
            return s
        m = re.search(r"(?:track[:/])(\d+)", s)
        if m:
            return m.group(1)
        m = re.search(r"(\d+)$", s)
        return m.group(1) if m else None


# -------- платформа --------
async def async_setup_platform(
    hass: HomeAssistant,
    config,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Yandex Music now playing sensor."""
    name = config[CONF_NAME]
    interval_td = config.get(CONF_SCAN_INTERVAL)
    upd_secs = config.get("update_interval")
    if upd_secs is not None:
        interval_td = timedelta(seconds=int(upd_secs))

    token = config.get(CONF_TOKEN)
    username = config.get(CONF_USERNAME)
    password = config.get(CONF_PASSWORD)
    use_ynison = bool(config.get("ynison", True))
    ynison_mode = config.get("ynison_mode", "put")
    if ynison_mode != "put":
        _LOGGER.warning("ynison_mode=%s игнорируется; используем 'put'", ynison_mode)

    # список media_player.* от AlexxIT (строкой через запятую или списком)
    station_entities_cfg = config.get("station_entities")
    station_entities: list[str] = []
    if isinstance(station_entities_cfg, str):
        station_entities = [e.strip() for e in station_entities_cfg.split(",") if e.strip()]
    elif isinstance(station_entities_cfg, (list, tuple)):
        station_entities = [str(e).strip() for e in station_entities_cfg if str(e).strip()]

    # --- опциональный Last.fm ---
    lastfm_cfg = config.get("lastfm")
    scrobbler: Optional[LastFmScrobbler] = None
    if lastfm_cfg and lastfm_cfg.get("enabled", True):
        try:
            scrobbler = LastFmScrobbler(
                hass=hass,
                api_key=lastfm_cfg["api_key"],
                api_secret=lastfm_cfg["api_secret"],
                session_key=lastfm_cfg["session_key"],
                min_scrobble_percent=lastfm_cfg.get("min_scrobble_percent", 50),
                min_scrobble_seconds=lastfm_cfg.get("min_scrobble_seconds", 240),
            )
            _LOGGER.info("Last.fm scrobbling is enabled")
        except Exception as e:
            _LOGGER.warning("Last.fm init failed: %r", e)
            scrobbler = None

    # --- аутентификация в Yandex Music SDK ---
    def _build_client_and_token():
        from yandex_music import Client  # type: ignore

        if token:
            _LOGGER.debug("Auth: using X-Yandex-Music-Token (length=%s)", len(token))
            client = Client(token).init()
            token_final_local = token
        elif username and password:
            _LOGGER.debug("Auth: using username/password (fetch token internally)")
            client = Client.from_credentials(username, password).init()
            token_final_local = getattr(client, "token", None)
        else:
            raise RuntimeError("Provide 'token' OR 'username'+'password'")

        try:
            me = client.me
            uid = getattr(getattr(me, "account", None), "uid", None)
            login = getattr(getattr(me, "account", None), "login", None)
            region = getattr(getattr(me, "account", None), "region", None)
            _LOGGER.debug("Auth OK: uid=%s login=%s region=%s", uid, login, region)
        except Exception as e:
            _LOGGER.warning("Auth check failed: cannot read client.me: %s", e)

        return client, token_final_local

    client, token_final = await hass.async_add_executor_job(_build_client_and_token)
    if not token_final:
        token_final = getattr(client, "token", None)
    if not token_final:
        raise RuntimeError("Cannot obtain OAuth token for Ynison")

    # общий доступ для switch и прочего
    data = hass.data.setdefault(DOMAIN, {})
    data["client"] = client
    data["token"] = token_final
    data["current_track_id"] = None

    # основной сенсор
    entity = YandexMusicNowPlayingSensor(
        hass=hass,
        client=client,
        token=token_final,
        name=name,
        poll_interval=interval_td,
        enable_ynison=use_ynison,
        station_entities=station_entities,
        scrobbler=scrobbler,
    )
    async_add_entities([entity], False)

    # сервисы лайка
    async def async_like_current(call: ServiceCall): await entity.async_like_current()
    async def async_unlike_current(call: ServiceCall): await entity.async_unlike_current()
    async def async_toggle_like(call: ServiceCall): await entity.async_toggle_like()

    hass.services.async_register(DOMAIN, "like_current", async_like_current)
    hass.services.async_register(DOMAIN, "unlike_current", async_unlike_current)
    hass.services.async_register(DOMAIN, "toggle_like", async_toggle_like)


class YandexMusicNowPlayingSensor(SensorEntity):
    _attr_icon = "mdi:music-note"

    def __init__(
        self,
        hass: HomeAssistant,
        client: Any,
        token: str,
        name: str,
        poll_interval: timedelta,
        enable_ynison: bool,
        station_entities: list[str],
        scrobbler: Optional[LastFmScrobbler],
    ):
        self.hass = hass
        self._client = client
        self._token = token
        self._attr_name = name
        self._attr_native_value = None
        self._attrs: dict = {}
        self._last_track_id = None
        self._scan_interval = poll_interval
        self._enable_ynison = enable_ynison
        self._ynison: Optional[YnisonWatcher] = None
        self._last_push: Optional[dict] = None

        # Слушаем эти media_player.* (AlexxIT)
        self._station_entities = list(dict.fromkeys(station_entities))  # уникальные, порядок сохранён
        self._unsub_station = None
        self._last_resolve_key: Optional[str] = None  # чтобы не спамить поиск

        # Last.fm
        self._scrobbler = scrobbler

        # --- ДЛЯ ОТКАТА ПРИ НЕИГРАЮЩЕЙ КОЛОНКЕ ---
        self._last_data: Optional[dict] = None
        self._last_data_source: Optional[str] = None

    @property
    def should_poll(self) -> bool:
        # Если есть push (ynison) — сенсор сам обновляется по push,
        # но мы всё равно слушаем media_player через подписку (это не poll).
        return not self._enable_ynison

    @property
    def extra_state_attributes(self):
        return self._attrs

    @property
    def native_value(self):
        return self._attr_native_value

    @property
    def entity_picture(self) -> str | None:
        return self._attrs.get("cover")

    @property
    def unique_id(self) -> str:
        return "yandex_music_nowplaying_account_default"

    @property
    def scan_interval(self) -> timedelta:
        return self._scan_interval

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self._enable_ynison:
            self._ynison = YnisonWatcher(
                hass=self.hass,
                token=self._token,
                yaclient=self._client,
                on_update=self._handle_push_update,
            )
            self._ynison.start()

        # первичная загрузка лайков (асинхронно, без блокировок)
        self.hass.async_create_task(self._async_refresh_likes_cache())

        # подписка на изменения медиаплееров AlexxIT
        if self._station_entities:
            self._unsub_station = async_track_state_change(
                self.hass, self._station_entities, self._on_station_state
            )
            _LOGGER.info("Listening media_player(s): %s", ", ".join(self._station_entities))

    async def async_will_remove_from_hass(self) -> None:
        if self._ynison:
            await self._ynison.stop()
        if self._unsub_station:
            try:
                self._unsub_station()
            except Exception:
                pass
        if self._scrobbler:
            self._scrobbler.shutdown()

    # ----------------- PUSH из Ynison -----------------
    @callback
    def _handle_push_update(self, data: Optional[dict]) -> None:
        self._apply_update(data, source="ynison", push=True)

    # ----------------- PULL-фолбэк (очереди) -----------------
    async def async_update(self) -> None:
        if self._enable_ynison and self._last_push is not None:
            return
        data = await self.hass.async_add_executor_job(self._fetch_now_playing_pull)
        self._apply_update(data, source="queues", push=False)

    # ----------------- Применение обновления -----------------
    def _apply_update(self, data: Optional[dict], source: str, *, push: bool) -> None:
        self._last_push = data if source == "ynison" else self._last_push
        if not data:
            self._attr_native_value = None
            self._attrs = {}
            self._last_track_id = None
            shared = self.hass.data.setdefault(DOMAIN, {})
            shared["current_track_id"] = None
            if push or self.entity_id is not None:
                self.async_write_ha_state()
            return

        self._last_track_id = data.get("track_id")
        title = data.get("title") or ""
        artists = data.get("artists") or ""
        self._attr_native_value = f"{artists} — {title}" if artists else title

        liked = self._is_liked(self._last_track_id)

        self._attrs = {
            "title": title,
            "artists": artists,
            "album": data.get("album"),
            "track_id": self._last_track_id,
            "cover": data.get("cover"),
            "duration_ms": data.get("duration_ms"),
            "explicit": data.get("explicit"),
            "context_type": data.get("context_type"),
            "queue_id": data.get("queue_id"),
            "source": source,
            "liked": liked,
        }

        # запомним последнюю валидную информацию сенсора для возможного отката
        if source in ("queues", "ynison"):
            try:
                self._last_data = dict(data)
            except Exception:
                self._last_data = data
            self._last_data_source = source

        # шарим текущий трек для switch
        shared = self.hass.data.setdefault(DOMAIN, {})
        shared["current_track_id"] = self._last_track_id

        if push or self.entity_id is not None:
            self.async_write_ha_state()

        # Last.fm now playing / планирование scrobble (если включён)
        if self._scrobbler:
            self._scrobbler.on_track_update(
                artist=artists,
                title=title,
                album=self._attrs.get("album"),
                duration_ms=self._attrs.get("duration_ms"),
                progress_sec=None,  # из очередей прогресса нет
            )

    # ----------------- PULL: очереди Я.Музыки -----------------
    def _fetch_now_playing_pull(self) -> dict | None:
        try:
            queues = self._client.queues_list()
            if not queues:
                _LOGGER.debug("queues_list(): empty")
                return None

            try:
                queues_sorted = sorted(
                    queues, key=lambda q: getattr(q, "modified", ""), reverse=True
                )
            except Exception:
                queues_sorted = queues

            for qi in queues_sorted:
                qid = getattr(qi, "id", None) or getattr(qi, "queue_id", None)
                q = None
                if hasattr(qi, "fetch_queue"):
                    try:
                        q = qi.fetch_queue()
                    except Exception:
                        q = None
                if q is None and qid:
                    try:
                        q = self._client.queue(qid)
                    except Exception:
                        q = None
                if not q:
                    continue

                current_index = getattr(q, "current_index", -1)
                if current_index is None or current_index < 0:
                    continue

                try:
                    tid = q.get_current_track()
                except Exception:
                    tid = None
                if not tid:
                    continue

                track_id = (
                    getattr(tid, "id", None) or getattr(tid, "track_id", None) or tid
                )
                if not track_id:
                    continue

                tr_list = self._client.tracks(track_id)
                track = (
                    tr_list[0]
                    if isinstance(tr_list, list) and tr_list
                    else (tr_list if tr_list else None)
                )
                if not track:
                    continue

                artists = (
                    ", ".join([a.name for a in getattr(track, "artists", [])])
                    if getattr(track, "artists", None)
                    else None
                )
                album_title = (
                    getattr(track.albums[0], "title", None)
                    if getattr(track, "albums", None)
                    else None
                )
                cover = None
                cover_uri = getattr(track, "cover_uri", None)
                if cover_uri:
                    uri = cover_uri.replace("%%", "300x300")
                    cover = uri if uri.startswith("http") else f"https://{uri}"
                context_type = getattr(getattr(q, "context", None), "type", None)

                return {
                    "title": getattr(track, "title", None),
                    "artists": artists,
                    "album": album_title,
                    "track_id": getattr(track, "id", None),
                    "cover": cover,
                    "duration_ms": getattr(track, "duration_ms", None),
                    "explicit": getattr(track, "explicit", None),
                    "context_type": context_type,
                    "queue_id": getattr(q, "id", None),
                }
            return None
        except Exception as e:
            _LOGGER.debug("Yandex Music pull failed: %s", e)
            return None

    # ----------------- Подписка на media_player.* (YandexStation) -----------------
    @callback
    def _on_station_state(self, entity_id: str, old_state: Optional[State], new_state: Optional[State]) -> None:
        if new_state is None:
            return
        attrs = new_state.attributes or {}
        state = new_state.state  # 'playing', 'paused', 'idle', ...

        # ---- Новая логика: если колонка НЕ играет — не затираем сенсор, а откатываемся к последнему валидному источнику ----
        if state != "playing":
            _LOGGER.debug("Station '%s' not playing (state=%s) — keep previous sensor data", entity_id, state)
            if self._last_data is not None:
                # мгновенно восстановим актуальные данные от Ynison/queues
                self._apply_update(dict(self._last_data), source=(self._last_data_source or "queues"), push=True)
            return

        # Извлекаем базовые поля
        title = (
            attrs.get("media_title")
            or attrs.get("title")
            or (attrs.get("player_state") or {}).get("title")
            or ""
        )
        artist = (
            attrs.get("media_artist")
            or attrs.get("subtitle")
            or attrs.get("artist")
            or (attrs.get("player_state") or {}).get("subtitle")
            or ""
        )
        album = (
            attrs.get("media_album_name")
            or attrs.get("album")
            or (attrs.get("player_state") or {}).get("album")
            or None
        )
        cover = (
            attrs.get("entity_picture")
            or attrs.get("media_image_url")
            or attrs.get("media_image")
            or (attrs.get("player_state") or {}).get("image")
            or None
        )
        progress = attrs.get("media_position") or (attrs.get("player_state") or {}).get("position_ms")
        duration = attrs.get("media_duration") or (attrs.get("player_state") or {}).get("duration_ms")

        # Пытаемся вытащить track_id из media_content_id
        content_id = (
            attrs.get("media_content_id")
            or attrs.get("content_id")
            or (attrs.get("player_state") or {}).get("id")
            or (attrs.get("player_state") or {}).get("playable_id")
        )
        track_id = self._normalize_track_id(content_id) or self._normalize_track_id(
            attrs.get("track_id") or attrs.get("yandex_music_track_id") or attrs.get("yandex_track_id")
        )

        # Обновляем видимую строку
        if title or artist:
            self._attr_native_value = f"{artist} — {title}" if artist and title else (title or artist)
        else:
            self._attr_native_value = None

        # Базовые атрибуты
        self._attrs.update({
            "title": title or None,
            "artists": artist or None,
            "album": album,
            "cover": cover,
            "player_state": state,
            "media_position": progress,
            "media_duration": duration,
            "source": "media_player",
            "station_entity": entity_id,
        })

        # Если у нас есть track_id — используем сразу
        if track_id:
            self._last_track_id = track_id
            self.hass.data.setdefault(DOMAIN, {})["current_track_id"] = self._last_track_id
            # отметка лайка из кэша
            liked = self._is_liked(self._last_track_id)
            self._attrs["track_id"] = self._last_track_id
            self._attrs["liked"] = liked

            # обложку и альбом дотянем, если нужно
            if not cover:
                self.hass.async_create_task(self._async_enrich_track_meta(track_id))

        else:
            # Нет id — резолвим в фоне по artist+title (без блокировок)
            key = f"{artist}@@{title}"
            if key and key != self._last_resolve_key and (artist or title):
                self._last_resolve_key = key
                self.hass.async_create_task(self._async_resolve_station_track(title, artist, album, cover))

        if self.entity_id is not None:
            self.async_write_ha_state()

        # Last.fm: передаём апдейт (с прогрессом и длительностью)
        if self._scrobbler:
            # progress может быть в секундах (media_position) или в мс (position_ms)
            prog_sec = None
            if progress is not None:
                try:
                    p = float(progress)
                    prog_sec = p / 1000.0 if p > 10000 else p
                except Exception:
                    prog_sec = None

            dur_ms = None
            if duration is not None:
                try:
                    d = float(duration)
                    dur_ms = int(d * 1000) if d < 10000 else int(d)
                except Exception:
                    dur_ms = None

            self._scrobbler.on_track_update(
                artist=artist,
                title=title,
                album=album,
                duration_ms=dur_ms,
                progress_sec=prog_sec,
            )

    @staticmethod
    def _normalize_track_id(val: Any) -> Optional[str]:
        if not val:
            return None
        s = str(val)
        if s.isdigit():
            return s
        # частые форматы: "track:123456", "music/track/123456", ".../track/123456?..."
        m = re.search(r"(?:track[:/])(\d+)", s)
        if m:
            return m.group(1)
        m = re.search(r"(\d+)(?:\D*$)", s)
        return m.group(1) if m else None

    # ----------------- лайки: асинхронная подгрузка кэша и неблокирующая проверка -----------------
    async def _async_refresh_likes_cache(self) -> None:
        """Загрузить все лайки в executor и сохранить в кэш (без блокировок)."""
        def _load_likes():
            try:
                return self._client.users_likes_tracks()
            except Exception:
                return None

        likes = await self.hass.async_add_executor_job(_load_likes)
        new_set: set[str] = set()
        if likes and getattr(likes, "tracks", None):
            for t in likes.tracks:
                tid = getattr(t, "id", None) or getattr(t, "track_id", None)
                if tid is not None:
                    new_set.add(str(tid))
        _likes_cache_set(self.hass, new_set)

    def _is_liked(self, track_id: Any) -> Optional[bool]:
        if not track_id:
            return None
        try:
            liked_set, ts = _likes_cache_get(self.hass)
            if liked_set is None or (time.time() - ts) > LIKES_TTL:
                # обновим кэш в фоне, вернём текущее значение
                self.hass.async_create_task(self._async_refresh_likes_cache())
                liked_set = liked_set or set()
            return str(track_id) in liked_set
        except Exception as e:
            _LOGGER.debug("is_liked check failed: %s", e)
            return None

    async def async_like_current(self) -> None:
        if not self._last_track_id:
            _LOGGER.warning("No current track to like")
            return

        def _like(track_id):
            self._client.users_likes_tracks_add(track_id)

        await self.hass.async_add_executor_job(_like, self._last_track_id)
        _likes_cache_touch_add(self.hass, self._last_track_id)
        self._attrs["liked"] = True
        if self.entity_id is not None:
            self.async_write_ha_state()
        _LOGGER.info("Liked track %s", self._last_track_id)

    async def async_unlike_current(self) -> None:
        if not self._last_track_id:
            _LOGGER.warning("No current track to unlike")
            return

        def _unlike(track_id):
            self._client.users_likes_tracks_remove(track_id)

        await self.hass.async_add_executor_job(_unlike, self._last_track_id)
        _likes_cache_touch_remove(self.hass, self._last_track_id)
        self._attrs["liked"] = False
        if self.entity_id is not None:
            self.async_write_ha_state()
        _LOGGER.info("Unliked track %s", self._last_track_id)

    async def async_toggle_like(self) -> None:
        liked = self._attrs.get("liked")
        if liked:
            await self.async_unlike_current()
        else:
            await self.async_like_current()

    # ----------------- Резолв/обогащение (фоном) -----------------
    async def _async_enrich_track_meta(self, track_id: str) -> None:
        """Дотянуть обложку/альбом/explicit по известному track_id (в executor)."""
        def _fetch():
            try:
                tr_list = self._client.tracks(track_id)
                return tr_list[0] if isinstance(tr_list, list) and tr_list else None
            except Exception:
                return None

        t = await self.hass.async_add_executor_job(_fetch)
        if not t:
            return
        try:
            if not self._attrs.get("cover") and getattr(t, "cover_uri", None):
                uri = t.cover_uri.replace("%%", "300x300")
                self._attrs["cover"] = uri if uri.startswith("http") else f"https://{uri}"
            if getattr(t, "albums", None):
                self._attrs["album"] = getattr(t.albums[0], "title", None) or self._attrs.get("album")
            self._attrs["duration_ms"] = getattr(t, "duration_ms", None)
            self._attrs["explicit"] = getattr(t, "explicit", None)
            # liked флажок обновим из кэша
            self._attrs["liked"] = self._is_liked(track_id)
            if self.entity_id is not None:
                self.async_write_ha_state()
        except Exception:
            pass

    async def _async_resolve_station_track(self, title: str, artist: str, album: Optional[str], cover: Optional[str]) -> None:
        """Фоновый резолв track_id по artist+title и обогащение метаданных."""
        def _search_and_fetch():
            try:
                query_parts = []
                if title: query_parts.append(title)
                if artist: query_parts.append(artist)
                q = " ".join(query_parts).strip()
                if not q:
                    return None, None
                res = self._client.search(q, type_="track")
                cand = None
                if getattr(res, "best", None) and getattr(res.best, "type", "") == "track":
                    cand = res.best.result
                elif getattr(res, "tracks", None) and getattr(res.tracks, "results", None):
                    cand = res.tracks.results[0]
                if not cand:
                    return None, None
                tid = getattr(cand, "id", None)
                if tid is None:
                    return None, None
                tr_list = self._client.tracks(tid)
                t = tr_list[0] if isinstance(tr_list, list) and tr_list else None
                return str(tid), t
            except Exception:
                return None, None

        track_id, t = await self.hass.async_add_executor_job(_search_and_fetch)

        if not track_id:
            self.hass.data.setdefault(DOMAIN, {})["current_track_id"] = None
            return

        self._last_track_id = track_id
        self.hass.data.setdefault(DOMAIN, {})["current_track_id"] = self._last_track_id
        self._attrs["track_id"] = track_id

        if t:
            try:
                if not cover and getattr(t, "cover_uri", None):
                    uri = t.cover_uri.replace("%%", "300x300")
                    self._attrs["cover"] = uri if uri.startswith("http") else f"https://{uri}"
                if getattr(t, "albums", None):
                    self._attrs["album"] = getattr(t.albums[0], "title", None) or self._attrs.get("album")
                self._attrs["duration_ms"] = getattr(t, "duration_ms", None)
                self._attrs["explicit"] = getattr(t, "explicit", None)
                self._attrs["liked"] = self._is_liked(self._last_track_id)
            except Exception:
                pass

        if self.entity_id is not None:
            self.async_write_ha_state()
