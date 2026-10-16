# 本地产物与冷归档导航

`outputs/` 仅保留历史运行产物、本地冷归档及恢复入口，不再新增实验或入库报告。已跟踪的历史JSON迁至 `reports/legacy/outputs/`，映射见[迁移清单](../plans/reference/PROJECT_LAYOUT_MIGRATION_20261016.json)。目录规则见 [项目地图](../docs/README.md) 和 [目录规范](../docs/FOLDER_STRUCTURE_RULES.md)。

## 2026-10-14 冷归档

已归档原路径：

| 原路径 | 本地归档包 |
| --- | --- |
| `output/epi_smoke/` | `cold_archive_20261014/epi_smoke.zip` |
| `output/n3_10_smoke_off/` | `cold_archive_20261014/n3_10_smoke_off.zip` |
| `output/n5_shadow_feed2/` | `cold_archive_20261014/n5_shadow_feed2.zip` |
| `output/ds-bot-0.1.2.tgz` | `cold_archive_20261014/ds-bot-0.1.2.tgz.zip` |

原件已移除；历史文件中的原路径不改写，需复算时先恢复。归档是本机副本，不是异地备份；Git 只保存清单，不保存 ZIP。

[逐文件恢复清单](../plans/reference/PROJECT_COLD_ARCHIVE_MANIFEST_20261014.json)记录包和文件的 SHA-256；[完成回执](../plans/reference/PROJECT_COLD_ARCHIVE_COMPLETION_20261014.json)记录删除与剩余产物分类。

## 恢复方法

在仓库根运行下列示例，只恢复 `epi_smoke`。其他包按表替换 `name`。脚本先核对包摘要、目标边界、原路径不存在及包内文件摘要，再解压到 `output/`；不覆盖已有实验。

```python
import hashlib
import json
import zipfile
from pathlib import Path

root = Path.cwd().resolve()
name = "output/epi_smoke"
manifest = json.loads(
    (root / "plans/reference/PROJECT_COLD_ARCHIVE_MANIFEST_20261014.json")
    .read_text(encoding="utf-8")
)
entry = next(item for item in manifest["entries"] if item["source"] == name)
target = root / name
if target.exists():
    raise RuntimeError("Restore target already exists")
archive = root / entry["archive"]
if hashlib.sha256(archive.read_bytes()).hexdigest() != entry["archive_sha256"]:
    raise RuntimeError("Archive checksum mismatch")
with zipfile.ZipFile(archive) as bundle:
    if set(bundle.namelist()) != {item["path"] for item in entry["files"]}:
        raise RuntimeError("Archive member list mismatch")
    for item in entry["files"]:
        destination = (root / "output" / item["path"]).resolve()
        if not destination.is_relative_to(root / "output") or destination.exists():
            raise RuntimeError("Unsafe or occupied restore destination")
        if hashlib.sha256(bundle.read(item["path"])).hexdigest() != item["sha256"]:
            raise RuntimeError("File checksum mismatch")
    bundle.extractall(root / "output")
```

其余本地产物保留原位置。生成于活动回归的结果、被测试默认路径依赖的目录和冻结证据，不按文件年龄自动清理。
