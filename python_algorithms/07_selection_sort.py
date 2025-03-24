def swap(i, j, array):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def selection_sort(array):
    for i in range(len(array)):
        min_pos = i
        for j in range(i + 1, len(array)):
            if (array[j] < array[min_pos]):
                min_pos = j
        if (i != min_pos):
            swap(i, min_pos, array)
    return

if __name__ == "__main__":
    array = [9, 15, 18, 3, 11, 6, 5]
    print("Array before sorted:", array)
    selection_sort(array)
    print("Array after sorted: ", array)
