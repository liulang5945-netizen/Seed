# 19.2 负方向脚本归档（D16-21 / D16-19 判废）

本目录只放**已被判废、不再在当前轨道上执行**的 19.2 系列脚本，保留作为证据，不得重新纳入调度或改造后复用。

| 脚本 | 所属方向 | 判废依据 | 证据 |
|---|---|---|---|
| `probe_taiji_package19_2n_twin_pool.py` | 19.2n 同池孪生（记录侧字节置换） | **D16-21**：记录侧字节置换轴作废（选择侧只读记录描述、查询侧只有问题句、两词表整串互斥）；19.2n 实测 `twin_not_constructible`（池内可构造孪生 0/384） | `reports/taiji_package19_2n_twin_pool_20261105.json`；账本 §16.12 / §16.15 |
| `probe_taiji_package19_2p_key_separability.py` | 19.2p 候选 key 改整行（角色可分辨性） | **D16-19**（池外负样本不受理）＋ **D16-21**；19.2p 判 `proposal_not_admissible`（2a 非统一分离器 81/96、2b/3 测量集为空） | `reports/taiji_package19_2p_key_separability_20261105.json`；账本 §16.14 / §16.15 |

19.2m 无独立脚本（其读数由 19.2n/19.2p 复核覆盖）。三者的方向已整体并入 D16-21，当前轨道不再引用。
