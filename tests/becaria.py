from memory_profiler import profile

@profile
def my_function():
    a = [1, 2, 3]
    print(sum(a))

if __name__ == '__main__':
    my_function()
