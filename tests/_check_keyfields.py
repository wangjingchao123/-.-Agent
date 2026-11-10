# -*- coding: utf-8 -*-
import json
with open("taiji_keyframes.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print("meta:", data.get("meta"))
tpls = data["templates"]
print("template count:", len(tpls))
for t in tpls:
    keys = list(t.keys())
    vec_len = len(t.get("vector", []))
    print(f"  {t['template_name']}: keys={keys}, vector_len={vec_len}")