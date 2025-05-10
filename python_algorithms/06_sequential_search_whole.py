def sequential_search(target, array):
    found = False
    for i in range(len(array)):
        if (array[i] == target):
            found = True
            print("Find it, the index is: ", i)

    if(found == False):
        print("Can not find it.")
    return

if __name__ == "__main__":
    target = 12
    array = [3, 5, 6, 8, 11, 15, 20, 21]
    print("Array:", array)
    print("Target:", target)
    sequential_search(target, array)
