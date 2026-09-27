"""UTF-8 字节序列状态机——产品里唯一一份合法后继字节集的实现。

SPEC-R2-02（解码掩码产品化）：`Taiji.generate(utf8_strict=True)` 与语言器官的约束解码
（`language_alignment._constrained_generate`）都从这里取数，**不许两处各写一份**——
分叉的两份掩码会静默漂移，这正是本仓多次实测"读数依链路而定"的成因形态。

语义（与 `language_alignment`/`probe_taiji_cap0_byte_output` 修前的两份实现逐字节一致，
等价测试见 `tests/taiji_native/test_utf8_strict_generation.py`）：
* ``remaining == 0``（字符边界）⇒ 合法首字节：ASCII ``0x00..0x7F``，或 2/3/4 字节
  序列的引导字节 ``0xC2..0xF4``（``0xC0/0xC1`` 是 overlong，代理区与超范围在下界的
  收紧里一并排除）；
* ``remaining > 0``（字符中间）⇒ 续字节 ``0x80..0xBF``，并按首字节收紧标量值域界
  （E0/ED 的第二字节、F0/F4 的第二字节）。
"""

from __future__ import annotations

#: PLAN-R2-01：读出位置输入的维数——“还期望几个续字节”的 one-hot 宽度（0..3）。
UTF8_POSITION_DIM = 4


def utf8_allowed(remaining: int, lead: int) -> list[int]:
    """给定"还期望几个续字节"与当前字符的首字节，返回合法后继字节全集。"""

    if remaining == 0:
        return list(range(0x00, 0x80)) + list(range(0xC2, 0xF5))
    low, high = 0x80, 0xBF
    # A 3-byte sequence has two continuation bytes remaining after its
    # lead; the first one must enforce the E0/ED scalar-value bounds.
    if remaining == 2 and lead == 0xE0:
        low = 0xA0
    elif remaining == 2 and lead == 0xED:
        high = 0x9F
    # A 4-byte sequence has three continuation bytes remaining after its
    # lead; the first one must enforce the F0/F4 scalar-value bounds.
    elif remaining == 3 and lead == 0xF0:
        low = 0x90
    elif remaining == 3 and lead == 0xF4:
        high = 0x8F
    return list(range(low, high + 1))


def advance_utf8(remaining: int, lead: int, symbol: int) -> tuple[int, int]:
    """消费一个字节后的新状态 ``(remaining, lead)``。"""

    if remaining == 0:
        if symbol < 0x80:
            return 0, 0
        if symbol < 0xE0:
            return 1, symbol
        if symbol < 0xF0:
            return 2, symbol
        return 3, symbol
    return remaining - 1, lead


def remaining_after(remaining: int, symbol: int) -> int:
    """只推进 DFA 的 ``remaining`` 分量（``lead`` 只收紧值域界，不参与位置推进）。

    PLAN-R2-01 的读出位置输入就取这个值（0..3 的 one-hot）。它**不另写一份判定**，
    而是委托给 ``advance_utf8``，保证与解码掩码用的是同一个状态机（本模块的设计纪律）。
    """

    return advance_utf8(int(remaining), 0, int(symbol))[0]


def trim_partial_tail(raw: bytes) -> bytes:
    """截掉结尾悬空的多字节序列前缀，使整串可解码（与 M1 仪器同法）。

    仪器版在"整串无一前缀可解码"时原样返回（对结构自洽的掩码输出永不触发）；
    本件把它改成返回空串——掩码输出若真到这里，说明状态机失步，
    交空串比交非法串诚实。
    """

    for cut in range(len(raw), max(0, len(raw) - 4), -1):
        try:
            raw[:cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return raw[:cut]
    return b""
