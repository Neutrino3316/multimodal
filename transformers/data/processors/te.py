def a():
    return b()

def b():
    return 'Hello', 2

if __name__ == '__main__':
    c , d = a()
    print(c,d)
    