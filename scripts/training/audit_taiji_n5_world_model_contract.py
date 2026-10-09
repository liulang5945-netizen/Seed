"""N5 §6 的"世界模型接口"合同普查（零算力，只读源码）。

**为什么要这台仪器**：09 规划 §2 N5 的痛点证据写着"WorldDynamicsLearner 零真实调用"，
而 §6 给了它一份四法合同（`observe(observation,state)→updated_state`／
`propose(goal,state,budget)→candidates`／`feedback(real_outcome,state)→update_record`／
`snapshot`＋`restore` 要能完整继续）。这句话长期是**散文**：谁都可以说"接口已具备"，
也可以说"还差一点"，而两种说法都没有读数。本件把它变成能红的读数。

**取法说清楚，不许被读成语义判定**：本器只做**方法名**与**引用点**的枚举，
即"这份件里有没有一个公开方法叫这个名字、生产链里有没有文件引用这个类"。
名字对上**不等于**语义对上（本仓踩过太多次"名字会伪装轴"）；反过来名字不在就等于合同缺位。
所以出件同时出版三样：逐动词的候选方法名与其签名、该类**全部**公开方法名（防我只查我猜的那几个）、
生产链引用点计数。

**rc 语义（别把 0 读成"合同齐"）**：`0`＝普查本身跑完了（无论合同齐不齐）；
`2`＝取法失败（文件缺、AST 解析不了、面为空）⇒ 那是"没算"，不是"没缺"。
合同齐不齐走 `contract_verdict` 字段，不看 rc。
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: §6 那份合同的四个动词 ⇒ 允许的公开方法名别名（写死在这里，扫描器不去"猜相近的名字"）。
CONTRACT_VERBS: dict[str, tuple[str, ...]] = {
    "observe": ("observe", "online_update"),
    "propose": ("propose",),
    "feedback": ("feedback", "record_schema_feedback"),
    "snapshot_restore": ("snapshot", "restore"),
}

#: 被普查的类与它所在的文件。
LEARNER_CLASS = "WorldDynamicsLearner"
LEARNER_FILE = "taiji/world_learning.py"

#: "生产链"的面：这些目录里的 .py 都算生产/训练链，逐个枚举后计数，不抽样。
PRODUCTION_FACES = ("taiji", "seed", "scripts/training", "api")


def _public_methods(path: Path, class_name: str) -> tuple[dict[str, str], list[str]]:
    """返回 ``{方法名: 签名字符串}`` 与该类里全部公开方法名；私有方法单独排除在候选之外。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            found: dict[str, str] = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    found[item.name] = ast.unparse(item.args)
            return found, sorted(found)
    raise ValueError(f"class {class_name} not found in {path}")


def _signature_for(method: str, table: dict[str, str]) -> str:
    return table.get(method, "<absent>")


def _call_sites(root: Path, faces: tuple[str, ...], token: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for face in faces:
        directory = root / face
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            hits = path.read_text(encoding="utf-8", errors="replace").count(token)
            if hits:
                counts[path.relative_to(root).as_posix()] = hits
    return counts


def audit(root: Path) -> dict[str, Any]:
    learner = root / LEARNER_FILE
    if not learner.is_file():
        raise FileNotFoundError(learner)
    table, public_names = _public_methods(learner, LEARNER_CLASS)
    verbs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for verb, aliases in CONTRACT_VERBS.items():
        present = [name for name in aliases if name in table]
        verbs[verb] = {
            "matched_methods": present,
            "signatures": {name: _signature_for(name, table) for name in present},
            "looked_for": list(aliases),
        }
        if not present:
            missing.append(verb)
    sites = _call_sites(root, PRODUCTION_FACES, LEARNER_CLASS)
    trainer_face = "scripts/training/train_seed_corpus.py"
    return {
        "format": "taiji-n5-world-model-contract-census-v1",
        "learner_file": LEARNER_FILE,
        "learner_class": LEARNER_CLASS,
        "public_method_count": len(public_names),
        "public_methods": public_names,
        "contract": verbs,
        "missing_verbs": missing,
        "contract_verdict": "contract_complete" if not missing else "contract_incomplete",
        "reference_sites": sites,
        "reference_site_total": sum(sites.values()),
        "corpus_trainer_references_learner": int(sites.get(trainer_face, 0)),
        #: 取法声明：名字匹配不代替语义判定（见 docstring）。
        "reading_limit": "method-name match only; semantic equivalence needs a signature read",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--out-report", type=Path, required=True)
    #: 面可换 ⇒ 拒绝支能在 `tmp_path` 上实走（钉死仓根的话，"没算"那一支永远测不到）。
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)

    try:
        payload = audit(args.root)
    except (FileNotFoundError, SyntaxError, ValueError) as error:
        #: 取法失败必须响亮：那既不是"合同齐"也不是"合同缺"，而是"没算"。
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(
            json.dumps(
                {
                    "format": "taiji-n5-world-model-contract-census-v1",
                    "status": "census_failed",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 2
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
