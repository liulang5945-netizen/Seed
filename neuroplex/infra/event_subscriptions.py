"""引擎间事件订阅注册（Deep Coupling 挂载点）。

启动时由 ``api.app._load_model_background`` 调用。当前注册：

- **事件审计**：通配订阅把所有事件追加写入 ``logs/events.jsonl``，
  给公测期提供可回放的生命/改进事件轨迹（单行 JSON，崩溃不丢历史）。

后续引擎间联动（如 improvement_proposal → 训练建议）在此追加订阅，
保持“订阅接线集中一处、发布方不感知消费方”的解耦约定。
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTERED = False

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_EVENT_LOG = _LOG_DIR / "events.jsonl"


def _audit_sink(message: dict) -> None:
    """通配订阅者：事件落盘审计轨迹（写失败只记日志，不影响事件流）。"""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_EVENT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"EventBus audit sink failed: {exc}")


def register_all_subscriptions() -> None:
    """幂等注册所有引擎间订阅。"""
    global _REGISTERED
    if _REGISTERED:
        return
    from neuroplex.infra.events import get_event_bus

    bus = get_event_bus()
    bus.subscribe("*", _audit_sink)
    _REGISTERED = True
    logger.info("EventBus subscriptions registered (audit sink active)")
