import sys
sys.path.insert(0, ".")
import v03_camera_dtw_test as m

lib = m.load_templates_only(m.TEMPLATE_FILE)
print(f"IMPORT_OK  templates={len(lib['templates'])}")
print(f"camera_fn={m.run_camera_test.__name__}, extract={m.extract_frame_33.__name__}")