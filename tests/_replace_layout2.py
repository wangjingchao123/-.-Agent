import ast

fp = r'd:\豆包\老年健康\app.py'
with open(fp, encoding='utf-8') as f:
    lines = f.readlines()

# ---- 锚点验证（1-based 行号）----
assert 'with gr.Row():' in lines[923], f'924锚点失败: {lines[923][:60]}'
assert 'outputs=[report_out])' in lines[969], f'970锚点失败: {lines[969][:60]}'
assert 'footer { display: none' in lines[912], f'913锚点失败: {lines[912][:60]}'
print('锚点验证通过')

# ---- 读取新布局 ----
with open(r'd:\豆包\老年健康\tests\_new_layout2.txt', encoding='utf-8') as f:
    new_layout = f.read()
if not new_layout.endswith('\n'):
    new_layout += '\n'

# ---- CSS追加块 ----
css_add = """
/* ---------- 按钮占满整栏 + 视频区压缩高度 ---------- */
.full-btn {
    width: 100% !important;
    min-height: 58px !important;
    display: block !important;
}
.left-col  { flex: 1 1 0 !important; }
.right-col { flex: 1.2 1 0 !important; }
.card-ui .image-container { height: 350px !important; }
.card-ui .image-container img { object-fit: contain !important; }
.card-ui .video-container { height: 240px !important; }

"""

# ---- 从后往前替换 ----
# 1) 布局 924-970（0-based [923:970]）
lines = lines[:923] + [new_layout] + lines[970:]
# 2) CSS追加：在913行footer前插入（0-based 912前）
lines = lines[:912] + [css_add] + lines[912:]

with open(fp, 'w', encoding='utf-8') as f:
    f.writelines(lines)

ast.parse(open(fp, encoding='utf-8').read())
print(f'替换完成，AST OK，总行数: {len(lines)}')