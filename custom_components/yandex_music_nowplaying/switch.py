import time
import logging
from datetime import timedelta
from typing import Any, Optional

import voluptuous as vol

from homeassistant.components.switch import PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_NAME, CONF_SCAN_INTERVAL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

DOMAIN = "yandex_music_nowplaying"
DEFAULT_NAME = "Yandex Music Like (current)"
LIKES_TTL = 60  # сек

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_SCAN_INTERVAL, default=timedelta(seconds=10)): cv.time_period,
    }
)

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

async def async_setup_platform(
    hass: HomeAssistant,
    config,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Yandex Music like/unlike switch for current track."""
    name = config.get(CONF_NAME, DEFAULT_NAME)
    interval_td = config.get(CONF_SCAN_INTERVAL)

    # Пытаемся взять client, если сенсор уже загрузился
    data = hass.data.setdefault(DOMAIN, {})
    client = data.get("client")  # может быть None на старте — это нормально

    entity = YandexMusicLikeSwitch(hass=hass, client=client, name=name, interval_td=interval_td)
    async_add_entities([entity], False)

class YandexMusicLikeSwitch(SwitchEntity):
    _attr_icon = "mdi:heart"

    def __init__(self, hass: HomeAssistant, client: Any, name: str, interval_td: timedelta):
        self.hass = hass
        self._client = client
        self._attr_name = name
        self._scan_interval = interval_td
        self._is_on: Optional[bool] = None
        self._current_track_id: Any = None

    @property
    def unique_id(self) -> str:
        return "yandex_music_like_current_account_default"

    @property
    def should_poll(self) -> bool:
        return True

    @property
    def available(self) -> bool:
        return self._client is not None and self._current_track_id is not None

    @property
    def is_on(self) -> Optional[bool]:
        return self._is_on

    @property
    def scan_interval(self) -> timedelta:
        return self._scan_interval

    async def async_update(self) -> None:
        if self._client is None:
            self._client = self.hass.data.get(DOMAIN, {}).get("client")

        data = self.hass.data.setdefault(DOMAIN, {})
        self._current_track_id = data.get("current_track_id")

        if self._client is None or not self._current_track_id:
            self._is_on = None
            return

        self._is_on = await self.hass.async_add_executor_job(self._is_liked, self._current_track_id)

    def _is_liked(self, track_id: Any) -> Optional[bool]:
        try:
            liked_set, ts = _likes_cache_get(self.hass)
            if liked_set is None or (time.time() - ts) > LIKES_TTL:
                likes = self._client.users_likes_tracks()
                new_set: set[str] = set()
                if likes and getattr(likes, "tracks", None):
                    for t in likes.tracks:
                        tid = getattr(t, "id", None) or getattr(t, "track_id", None)
                        if tid is not None:
                            new_set.add(str(tid))
                _likes_cache_set(self.hass, new_set)
                liked_set = new_set
            return str(track_id) in liked_set
        except Exception as e:
            _LOGGER.debug("switch is_liked failed: %s", e)
            return None

    async def async_turn_on(self, **kwargs) -> None:
        if self._client is None or not self._current_track_id:
            _LOGGER.warning("No current track to like")
            return

        def _like(track_id):
            self._client.users_likes_tracks_add(track_id)

        await self.hass.async_add_executor_job(_like, self._current_track_id)
        _likes_cache_touch_add(self.hass, self._current_track_id)
        self._is_on = True
        if self.entity_id is not None:
            self.async_write_ha_state()
        _LOGGER.info("Liked track %s (via switch)", self._current_track_id)

    async def async_turn_off(self, **kwargs) -> None:
        if self._client is None or not self._current_track_id:
            _LOGGER.warning("No current track to unlike")
            return

        def _unlike(track_id):
            self._client.users_likes_tracks_remove(track_id)

        await self.hass.async_add_executor_job(_unlike, self._current_track_id)
        _likes_cache_touch_remove(self.hass, self._current_track_id)
        self._is_on = False
        if self.entity_id is not None:
            self.async_write_ha_state()
        _LOGGER.info("Unliked track %s (via switch)", self._current_track_id)
