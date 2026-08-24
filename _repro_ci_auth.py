import sys
import importlib.abc


class Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, names):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.names:
            raise ImportError(f"No module named '{root}' (simulated CI env)")
        return None


for mod in list(sys.modules):
    if mod.split(".")[0] in ("cryptography",):
        del sys.modules[mod]

sys.meta_path.insert(0, Blocker({"cryptography"}))

from neuroplex.core.security import AuthManager

try:
    auth = AuthManager()
    print("AuthManager constructed")
except Exception as exc:
    print(f"__new__ raised: {type(exc).__name__}: {exc}")

print("--- second call (singleton poisoned?) ---")
try:
    auth = AuthManager()
    print("has enabled:", hasattr(auth, "enabled"))
    print("enabled =", auth.enabled)
except Exception as exc:
    print(f"raised: {type(exc).__name__}: {exc}")
