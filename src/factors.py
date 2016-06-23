import sys

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

if __name__ == "__main__":
    arg = int(raw_input("Type number to get factors of: "))
    print factors(arg)
