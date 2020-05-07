from radon.metrics import *

with open('writeInAIR.py') as file :
    content =file.read()
    b = h_visit(content)

dictionary = b[0]._asdict()
for i in dictionary:
    print(i,':',dictionary[i])

