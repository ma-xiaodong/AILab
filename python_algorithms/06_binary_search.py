def binary_search(target, array):
    start = 0
    end = len(array) - 1
    found = False

    while start <= end:
        mid = start + (end - start) // 2
        if (array[mid] == target):
            found = True
            break
        elif (array[mid] > target):
            end = mid - 1
        else:
            start = mid + 1

    if(found == False):
        print("Can not find it.")
    else:
        print("Find it and the index is:", mid)

    return

if __name__ == "__main__":
    target = 5
    array = [3, 5, 6, 8, 11, 15, 20, 21]
    print("Array:", array)
    print("Target:", target)
    binary_search(target, array)
