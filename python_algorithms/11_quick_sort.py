def swap(i, j, array):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def split(low, high, array):
    if (low > high):
        return -1
    left = low + 1
    right = high

    # print("(", array[low], array[high], ")")
    while left <= right:
        if (array[left] > array[low]):
            swap(left, right, array)
            right = right - 1
        else:
            left = left + 1
    swap(low, left - 1, array)
    return left - 1

def quick_sort(array):
    todo_list = [[0, len(array) - 1]]
    i = 0

    while i < len(todo_list):
        mid = split(todo_list[i][0], todo_list[i][1], array)
        # print(array)
        if (mid != -1):
            todo_list.append([todo_list[i][0], mid - 1])
            todo_list.append([mid + 1, todo_list[i][1]])
        i = i + 1
    return
if __name__ == "__main__":
    array = [13, 7, 5, 1, 6, 9, 8, 11, 24, 21, 19, 23, 27, 26, 28]
    # array = [9, 15, 18, 3, 11, 6, 5]
    print("Array before sorted:", array)
    quick_sort(array)
    print("Array after sorted: ", array)
