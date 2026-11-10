# -*- coding: utf-8 -*-
"""精确替换 app.py 中 evaluate 函数的报告段落（仅此段，其它不动）"""
import ast

APP = r'd:\豆包\老年健康\app.py'
NEW = r'd:\豆包\老年健康\tests\_new_report.txt'

with open(APP, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(NEW, 'r', encoding='utf-8') as f:
    new_code = f.read().rstrip('\n')

# 定位起始行：包含"结构化 HTML 报告"或"面向老人的友好报告"注释的行
start = None
for i, ln in enumerate(lines):
    if '结构化 HTML 报告' in ln or '面向老人的友好报告' in ln:
        start = i
        break
assert start is not None, '找不到报告段落起始注释'

# 定位结束行：从 start 开始第一个内容为 """ 的行（report 字符串闭合）
end = None
for i in range(start + 1, len(lines)):
    if lines[i].rstrip('\r\n') == '"""':
        end = i
        break
assert end is not None, '找不到 report 字符串的闭合 """'

old_start_1based = start + 1
old_end_1based = end + 1

new_lines = lines[:start] + [new_code + '\n'] + lines[end + 1:]

with open(APP, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

ast.parse(open(APP, encoding='utf-8').read())
print('ast.parse OK')
print(f'旧报告段落行号范围：第 {old_start_1based} 行 ~ 第 {old_end_1based} 行（共 {old_end_1based - old_start_1based + 1} 行）')
print(f'新报告段落行数：{new_code.count(chr(10)) + 1} 行')
print(f'替换后文件总行数：{len(new_lines)}')