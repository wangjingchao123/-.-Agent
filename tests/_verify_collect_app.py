import sys
sys.path.insert(0, ".")
import taiji_collect as c

print("import OK")
print("CHOICES count:", len(c.CHOICES))
print("first:", c.CHOICES[0], "| last:", c.CHOICES[-1])
print("has 单鞭（一）:", "单鞭（一）" in c.CHOICES)
print("has 单鞭（二）:", "单鞭（二）" in c.CHOICES)
print("list_library lines:", len(c.list_library().splitlines()))
print("demo title:", c.demo.title)