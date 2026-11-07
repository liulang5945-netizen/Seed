# 禁改清单：在跑训练窗读取集（2026-11-08）

> 由 `scripts/maintenance/freeze_readset_20261108.py` 静态 AST 解析生成（不导入任何 taiji 模块、不加载权重、不训练）。
> 生成时间 `2026-11-07T22:12:02`，共 **25** 个文件；机器可读件 [manifest](../manifests/taiji_frozen_readset_20261108.json)。

## 约束

1. 下表任一文件在本次设计工作中**一个字节都不改**；改动只写进新文件（新设计模块/新文档），且新文件不得被 runner 或其 import 闭包引用。
2. 只要下表字节不变，对表外文件的任何编辑都不会影响已在内存中的训练进程，也不会改变续跑的身份判定输入。
3. 续跑身份判定的锚点：预注册 sha（runner 内常量）、runner sha（进程内自算）、检查点逐文件 sha——三者全部落在下表或检查点目录内。

## 文件与 sha256

| 角色 | 文件 | 字节 | sha256 |
|---|---|---:|---|
| imported module | `scripts/training/probe_taiji_f_expressive_capacity_20261108.py` | 47,319 | `3fb27e49df20b3061b43a25a051034fe43df957a9a481089caa1bbf710f819fb` |
| imported module | `scripts/training/probe_taiji_fec1_w1a_retest_20261108.py` | 24,119 | `5c1321f9ee2a6e29089e90b46387bd897d8838cf8f8a6b27b97d084ad8861942` |
| imported module | `scripts/training/probe_taiji_fec1_wv2_retest_20261108.py` | 21,759 | `04c0645dc9ca577993e3184a9696d0a301a5e45599e3e52e91ca8580a801b57d` |
| imported module | `scripts/training/probe_taiji_package20_2_credit_recovery.py` | 41,496 | `cd23ce2e06b4b6ad1455378f0d9df4f60bef98762893756f6302f3a709c04119` |
| imported module | `scripts/training/probe_taiji_package20_2s_f_information.py` | 30,406 | `ab9b73ec25a57e044eaa7b59f167e269e8bc520abe3eeb8e440a936836aca675` |
| imported module | `scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py` | 65,578 | `112c76ae117255bb5da7d60a46312bf82382430fc3a99ab8d7ce6a1f173d4095` |
| runner | `scripts/training/run_taiji_training_window_v2_20261108.py` | 24,272 | `f71014bfeb3a4dbe7cb111172f68c144b0d3a4f29d9e36ff50f0fe518555d3e5` |
| imported module | `taiji/__init__.py` | 51,815 | `9fdf2934790b96bc6db2bb14fa124b3f155d1b15e2106c03fbe0ff9e69953422` |
| imported module | `taiji/binding_copy.py` | 50,619 | `37f4885e8dc33a41f6f701e3d0e87daf15c126be735acceea02098e5e39c045c` |
| imported module | `taiji/binding_select.py` | 15,331 | `81667b9eed2db1e81b13d1f7cd83ec0d1570636b502ce74128e173a30813c846` |
| imported module | `taiji/candidate_manifest.py` | 4,579 | `6fc05ca4bcee45203594ba80448ed8f2c3d5d89e9df92ccf48c4ab23544db1f0` |
| imported module | `taiji/growth_credit.py` | 25,726 | `3c311b49b4ebefdb552dc52c6716941cf3ad5882a6bf55b295a1adc9769f3317` |
| imported module | `taiji/living_core.py` | 35,632 | `8c413795dc9112119a1fccf16256b1ffcd029e71df3fd9d95cc36d4050efed55` |
| imported module | `taiji/living_core_batch.py` | 23,343 | `b01d700b6d107fde66838e76b0bc587e99a0efb8e65d5dab193081e131787fc4` |
| imported module | `taiji/native_activity.py` | 11,690 | `3569a54ded60e3422497d59f97fb6dc005881a623697905727ba9376d4531e88` |
| imported module | `taiji/record_relation.py` | 20,802 | `6e38b889db732c405f41eefb86c34fe3c966b56b34bfe995c2d9c2a67f8a5f18` |
| imported module | `taiji/record_relation_core.py` | 38,298 | `8318dde5c3d3b3ff42a2c94e528aec7552363d4f9aef2c11ee95b8e868ef64a2` |
| imported module | `taiji/rule_learning_20_1_identity.py` | 4,897 | `91ae83b57ba2ce72d72c209fbff0ec55040b779059119779d23472d4602c8d79` |
| imported module | `taiji/rule_learning_20_1_net.py` | 23,798 | `0b0297eeb76805df011eaceb8d9c6a50cc012f43bd484dee0bd7663ba873a6fc` |
| imported module | `taiji/selection_routing.py` | 26,700 | `3b80ed033055ed9eb15afc386befae7d6b73ad829844c68a59a89f20b2aa5bde` |
| imported module | `taiji/selection_routing_checkpoint.py` | 15,647 | `606cc68cf1c5a628d3b91c494258c9f4b20084fa638eb88615e8e60dfcd5ce61` |
| imported module | `taiji/state_migration.py` | 24,262 | `89b9c123095c40891d95887e3d0ccb31cdd229f8b016471840c4bae8f612bbaa` |
| pre-registration | `plans/manifests/taiji_training_window_v2_prereg_20261108.json` | 3,439 | `b2db40018e89bffdd9b443b3b413cab5d1b479229c2d21834ee99d01b2366d12` |
| 20.4c r2 freeze manifest | `plans/manifests/taiji_package20_4c_r2_freeze_20261105.json` | 6,472 | `b7d940f7669b1adbea12ed8b940ee719f284eb050268ca32730eeebd40ec1998` |
| parent checkpoint (loaded by r2) | `output/taiji_package19_compare/28031-B_stop_lock-step300.pt` | 65,841,771 | `fe04fbfb14aa0390877385a4429f2a851396a6fe6acb3107bae0a5e4e331dfca` |

## 已知不一致（登记，不改）

- runner 文档字符串写「sha fdb975a7」，实际常量与预注册文件均为 `b2db40018e89…`；以常量/文件为准，文档字符串留待下一次允许改 runner 时修正。
