import sys
sys.path.insert(0, ".")
import taiji_template_lib as t

print("import OK")
print("templates:", len(t.saved_templates))
names = list(t.saved_templates.keys())
print("first3:", names[:3])
print("last:", names[-1])
print("TAIJI_24 count:", len(t.TAIJI_24))