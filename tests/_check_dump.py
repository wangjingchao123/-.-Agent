import io, os
p = os.path.join(os.environ['TEMP'], 'sa-dump.html')
t = io.open(p, encoding='utf-8', errors='ignore').read()
checks = {
    'demo-overlay存在': 'id="demo-overlay"' in t,
    'demo-overlay初始hidden': 'id="demo-overlay" class="demo-overlay" hidden' in t,
    'wait-confirm存在': 'id="wait-confirm"' in t,
    'wait-confirm初始hidden': 'id="wait-confirm" class="wait-confirm" hidden' in t,
    'btn-demo-close存在': 'id="btn-demo-close"' in t,
    'btn-wait-start存在': 'id="btn-wait-start"' in t,
    'taiji.js已引入': 'js/taiji.js' in t,
    'train.js已引入': 'js/train.js' in t,
    'emergency-overlay保留': 'id="emergency-overlay"' in t,
    'btn-emergency保留': 'id="btn-emergency"' in t,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print('ALL_OK' if all(checks.values()) else 'HAS_FAIL')