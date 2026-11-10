# -*- coding: utf-8 -*-
"""精确替换 app.py 中的 custom_css 字符串（仅此变量，其它不动）"""
import ast

APP = r'd:\豆包\老年健康\app.py'
NEW = r'd:\豆包\老年健康\tests\_new_css.txt'

with open(APP, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(NEW, 'r', encoding='utf-8') as f:
    new_css = f.read().rstrip('\n')

# 定位 custom_css = """ 起始行
start = None
for i, ln in enumerate(lines):
    if ln.startswith('custom_css = """'):
        start = i
        break
assert start is not None, '找不到 custom_css = """'

# 定位闭合 """ 行（从 start+1 开始找第一个内容为 """ 的行）
end = None
for i in range(start + 1, len(lines)):
    if lines[i].rstrip('\r\n') == '"""':
        end = i
        break
assert end is not None, '找不到 custom_css 的闭合 """'

old_start_1based = start + 1
old_end_1based = end + 1

# 组装新内容：custom_css = """\n + CSS + \n"""
new_block = 'custom_css = """\n' + new_css + '\n"""\n'
new_lines = lines[:start] + [new_block] + lines[end + 1:]

with open(APP, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# 验证语法
ast.parse(open(APP, encoding='utf-8').read())
print('ast.parse OK')
print(f'旧 custom_css 行号范围：第 {old_start_1based} 行 ~ 第 {old_end_1based} 行（共 {old_end_1based - old_start_1based + 1} 行）')
print(f'新 CSS 行数：{new_css.count(chr(10)) + 1} 行')
print(f'替换后文件总行数：{len(new_lines)}')