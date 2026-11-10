import ast

fp = r'd:\豆包\老年健康\app.py'
with open(fp, encoding='utf-8') as f:
    lines = f.readlines()

# 删除旧CSS残留：941-1008行（1-based）→ 0-based [940:1008]
assert 'display: inline-flex' in lines[940], f'941行不对: {lines[940][:50]}'
assert lines[1007].strip() == '}', f'1008行不对: {lines[1007][:50]}'
assert '按钮' in lines[1009], f'1010行不对: {lines[1009][:50]}'

new_lines = lines[:940] + lines[1008:]
with open(fp, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

ast.parse(open(fp, encoding='utf-8').read())
print(f'删除941-1008行完成，原{len(lines)}行→现{len(new_lines)}行，AST OK')