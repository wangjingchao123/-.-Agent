# -*- coding: utf-8 -*-
"""精确替换 app.py 中的 evaluate 函数（仅此函数，其它不动）"""
import ast

APP = r'd:\豆包\老年健康\app.py'
NEW = r'd:\豆包\老年健康\tests\_new_evaluate.txt'

with open(APP, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(NEW, 'r', encoding='utf-8') as f:
    new_func = f.read().rstrip('\n')

# 定位旧 evaluate 函数：起始行 = "def evaluate("，结束行 = 下一个顶层块之前
start = None
for i, ln in enumerate(lines):
    if ln.startswith('def evaluate('):
        start = i
        break
assert start is not None, '找不到 def evaluate('

end = None
for i in range(start + 1, len(lines)):
    s = lines[i].lstrip()
    if s.startswith('# ====') or (s.startswith('def ') and i != start):
        end = i
        break
assert end is not None, '找不到 evaluate 函数结束位置'

# 回退尾部的空行，保留两个空行分隔
while end > start + 1 and lines[end - 1].strip() == '':
    end -= 1

old_start_1based = start + 1
old_end_1based = end  # end 指向下一个块首行，旧函数实际占 [start, end-1]

new_lines = lines[:start] + [new_func + '\n', '\n', '\n'] + lines[end:]

with open(APP, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# 验证语法
ast.parse(open(APP, encoding='utf-8').read())
print('ast.parse OK')
print(f'旧函数行号范围：第 {old_start_1based} 行 ~ 第 {old_end_1based} 行（共 {old_end_1based - old_start_1based + 1} 行）')
print(f'新函数行数：{new_func.count(chr(10)) + 1} 行')
print(f'替换后文件总行数：{len(new_lines)}')