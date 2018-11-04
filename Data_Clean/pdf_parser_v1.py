from tika import parser
import re
import os


PAGE_SEPARATOR = re.compile(r'— PAGE [0-9]+ —')
REFERENCE = '4.0 REFERENCES'
INTRODUCTION = '1.0 INTRODUCTION'


filename = 'IRA_Plan_2-20220(final).pdf'
raw = parser.from_file(filename)
df = (raw['content'])

out_tmp = open('__tmp__.{0}.txt'.format(filename), 'w')
out_tmp.write(df)
out_tmp.close()


with open('__tmp__.{0}.txt'.format(filename), 'r') as f:
    processed = ""
    start_flag = False
    for line in f:
        line = line.strip()
        if line == '':
            continue

        if re.match(PAGE_SEPARATOR, line):
            continue

        if line == INTRODUCTION:
            start_flag = True

        if line == REFERENCE:
            break

        # line.replace(u'\U0000F0B7', "")
        if line.startswith(u'\U0000F0B7'):
            line = line[1:].strip()

        if start_flag:
            processed += str(line + '\r\n')

output = open('out_{0}.txt'.format(filename), 'w')
output.write(processed)
output.close()

os.remove('__tmp__.{0}.txt'.format(filename))