$ErrorActionPreference = "Stop"

python scripts/training/n31_guard.py N32.6.03 --seeds 20261121,20261122 --no-record
python scripts/training/n32_6_03_r1_joint_rep.py --smoke
python scripts/training/n32_6_03_r1_joint_rep.py
