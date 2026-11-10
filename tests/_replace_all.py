import ast

fp = r'd:\豆包\老年健康\app.py'
with open(fp, encoding='utf-8') as f:
    lines = f.readlines()

# ---- 锚点验证（1-based 行号）----
assert 'with gr.Row():' in lines[1052], f'1053锚点失败: {lines[1052][:60]}'
assert 'outputs=[report_out])' in lines[1098], f'1099锚点失败: {lines[1098][:60]}'
assert 'custom_css = """' in lines[765], f'766锚点失败: {lines[765][:60]}'
assert lines[1041].strip() == '"""', f'1042锚点失败: {lines[1041][:60]}'
assert '高级 CSS' in lines[764], f'765锚点失败: {lines[764][:60]}'
assert 'recent_status in' in lines[479], f'480锚点失败: {lines[479][:60]}'
print('锚点验证全部通过')

# ---- 读取新内容 ----
with open(r'd:\豆包\老年健康\tests\_new_css3.txt', encoding='utf-8') as f:
    new_css = f.read()
with open(r'd:\豆包\老年健康\tests\_new_layout.txt', encoding='utf-8') as f:
    new_layout = f.read()
if not new_layout.endswith('\n'):
    new_layout += '\n'

# ---- 从后往前替换 ----
# 1) 布局 1053-1099（0-based [1052:1099]）
lines = lines[:1052] + [new_layout] + lines[1099:]
# 2) CSS 765-1042（0-based [764:1042]）
lines = lines[:764] + [new_css] + lines[1042:]
# 3) 拦截判断 480（0-based 479）
lines[479] = '    if "严重心血管疾病" in chronic_list or recent_status == "是，存在关节剧痛/头晕":\n'

with open(fp, 'w', encoding='utf-8') as f:
    f.writelines(lines)

ast.parse(open(fp, encoding='utf-8').read())
print(f'三处替换完成，AST OK，总行数: {len(lines)}')