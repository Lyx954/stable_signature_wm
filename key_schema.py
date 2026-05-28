"""
Stable Signature 48-bit Key Schema v1
=====================================
结构化编码/解码：将 48-bit 水印密钥按字段划分，支持嵌入时编码元信息，
检测时解码还原。

Bit layout (48 bits total, MSB first):

  ┌─────────┬──────────┬──────────────┬──────────┬─────────┬──────────┐
  │ Version │  Model   │    Date      │   User   │ Content │ Reserved │
  │  4 bit  │  8 bit   │   12 bit     │  12 bit  │  4 bit  │  8 bit   │
  │  0 - 3  │  4 - 11  │  12 - 23     │ 24 - 35  │ 36 - 39 │ 40 - 47  │
  └─────────┴──────────┴──────────────┴──────────┴─────────┴──────────┘

Field details:
  Version   (4b):  Schema version, currently 1.  Range 0-15.
  Model     (8b):  Watermark-embedding model ID.  Range 0-255.
  Date      (12b): Days since 2025-01-01.          Range 0-4095 (~11 years).
  User      (12b): User/entity ID.                 Range 0-4095.
  Content   (4b):  Content category.               Range 0-15.
  Reserved  (8b):  Reserved / flags.               Range 0-255.

Usage:
  from key_schema import encode_key, decode_key, format_decoded

  # Encode
  key = encode_key(model_id=1, date_days=500, user_id=1, content_type=1)

  # Decode
  info = decode_key(key)
  print(info.model_name)  # Stable Diffusion 2.1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

BASE_DATE = date(2025, 1, 1)
SCHEMA_VERSION = 1

# ── Model table (watermark-embedding generative models) ────────────────
MODEL_TABLE: dict[int, dict] = {
    0:  {"name": "未指定",                   "type": "unknown"},
    1:  {"name": "Stable Diffusion 2.1",     "type": "image"},
    2:  {"name": "Stable Diffusion 2.0",     "type": "image"},
    3:  {"name": "Stable Diffusion XL 1.0",  "type": "image"},
    4:  {"name": "Stable Diffusion 3.5",     "type": "image"},
    5:  {"name": "SDXL Turbo",               "type": "image"},
    6:  {"name": "SD 3.5 Medium",            "type": "image"},
    7:  {"name": "Flux.1 Dev",               "type": "image"},
    8:  {"name": "Flux.1 Schnell",           "type": "image"},
    32: {"name": "ModelscopeT2V",            "type": "video"},
    33: {"name": "AnimateDiff",              "type": "video"},
}

# ── Content type table ─────────────────────────────────────────────────
CONTENT_TYPE_TABLE: dict[int, dict] = {
    0: {"name": "未分类",       "en": "Unclassified"},
    1: {"name": "纯AI生成",     "en": "Pure AI-generated"},
    2: {"name": "AI辅助编辑",   "en": "AI-assisted editing"},
    3: {"name": "深度伪造",     "en": "Deepfake"},
    4: {"name": "真实拍摄",     "en": "Real photograph"},
    5: {"name": "AI生成视频",   "en": "AI-generated video"},
    6: {"name": "AI生成音频",   "en": "AI-generated audio"},
    7: {"name": "AI生成文本",   "en": "AI-generated text"},
}

# ── User table ─────────────────────────────────────────────────────────
USER_TABLE: dict[int, str] = {
    0:   "系统自动",
    1:   "本地用户",
    100: "测试用户A",
    101: "测试用户B",
}


def encode_key(
    model_id: int = 0,
    date_days: Optional[int] = None,
    user_id: int = 0,
    content_type: int = 0,
    reserved: int = 0,
    version: int = SCHEMA_VERSION,
    target_date: Optional[date] = None,
) -> str:
    """Encode structured fields into a 48-bit binary key string."""
    if target_date is not None:
        date_days = (target_date - BASE_DATE).days
    elif date_days is None:
        date_days = (date.today() - BASE_DATE).days

    version      = max(0, min(15, version))
    model_id     = max(0, min(255, model_id))
    date_days    = max(0, min(4095, date_days))
    user_id      = max(0, min(4095, user_id))
    content_type = max(0, min(15, content_type))
    reserved     = max(0, min(255, reserved))

    value = 0
    value = (value << 4)  | version
    value = (value << 8)  | model_id
    value = (value << 12) | date_days
    value = (value << 12) | user_id
    value = (value << 4)  | content_type
    value = (value << 8)  | reserved

    return format(value, '048b')


@dataclass
class DecodedKey:
    """Structured result of decoding a 48-bit watermark key."""
    key_str: str
    version: int
    model_id: int
    model_name: str
    model_type: str
    date_days: int
    date: date
    user_id: int
    user_name: str
    content_type: int
    content_name: str
    reserved: int
    is_valid: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key_str,
            "version": self.version,
            "model": {"id": self.model_id, "name": self.model_name, "type": self.model_type},
            "date": {"days_since_2025": self.date_days, "date": self.date.isoformat()},
            "user": {"id": self.user_id, "name": self.user_name},
            "content": {"id": self.content_type, "name": self.content_name},
            "reserved": self.reserved,
            "is_valid": self.is_valid,
        }


def decode_key(key_str: str) -> DecodedKey:
    """Decode a 48-bit binary key string into structured fields."""
    key_str = key_str.strip()
    if len(key_str) != 48:
        raise ValueError(f"Key must be 48 bits, got {len(key_str)}")

    value = int(key_str, 2)

    reserved      = value & 0xFF;          value >>= 8
    content_type  = value & 0xF;           value >>= 4
    user_id       = value & 0xFFF;         value >>= 12
    date_days     = value & 0xFFF;         value >>= 12
    model_id      = value & 0xFF;          value >>= 8
    version       = value & 0xF

    model_info = MODEL_TABLE.get(model_id, {"name": f"未知({model_id})", "type": "unknown"})
    content_info = CONTENT_TYPE_TABLE.get(content_type, {"name": f"未知({content_type})"})
    user_name = USER_TABLE.get(user_id, f"用户#{user_id}")

    try:
        decoded_date = BASE_DATE + timedelta(days=date_days)
    except OverflowError:
        decoded_date = BASE_DATE

    return DecodedKey(
        key_str=key_str, version=version,
        model_id=model_id, model_name=model_info["name"], model_type=model_info["type"],
        date_days=date_days, date=decoded_date,
        user_id=user_id, user_name=user_name,
        content_type=content_type, content_name=content_info["name"],
        reserved=reserved,
        is_valid=(version == SCHEMA_VERSION),
    )


def format_decoded(info: DecodedKey, verbose: bool = False) -> str:
    """Format decoded key as human-readable string."""
    lines = [
        "=" * 55,
        f"  Schema Version : v{info.version} {'(valid)' if info.is_valid else '(unknown)'}",
        f"  Model          : {info.model_name} ({info.model_type})",
        f"  Date           : {info.date.isoformat()} (day {info.date_days} from 2025-01-01)",
        f"  User           : {info.user_name} (ID: {info.user_id})",
        f"  Content Type   : {info.content_name} (ID: {info.content_type})",
        f"  Reserved       : {info.reserved}",
        f"  Key            : {info.key_str}",
        "=" * 55,
    ]
    if verbose:
        lines.append(f"  Bits: [{info.key_str[0:4]}] [{info.key_str[4:12]}] [{info.key_str[12:24]}] [{info.key_str[24:36]}] [{info.key_str[36:40]}] [{info.key_str[40:48]}]")
    return "\n".join(lines)
