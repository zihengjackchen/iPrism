#!/usr/bin/python3

x1 = []
x2 = []

def parse_file(fname):
    with open (fname) as ifh:
        for line in ifh:
            y = set(line.split(':'))
            return y


x1 = parse_file("c1.txt")
x2 = parse_file("c2.txt")

print(x2-x1)
print(x1-x2)

