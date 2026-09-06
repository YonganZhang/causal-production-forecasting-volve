import json
import sys

from . import GaiaTeam

t = GaiaTeam()
if "--json" in sys.argv:
    print(json.dumps(t.evidence(), ensure_ascii=False, indent=2))
else:
    print(t.describe())
    print("=" * 72)
    print("关键实测事实")
    for k, v in t.evidence().items():
        print(f"\n[{k}]")
        print(json.dumps(v, ensure_ascii=False, indent=2) if isinstance(v, dict) else v)
