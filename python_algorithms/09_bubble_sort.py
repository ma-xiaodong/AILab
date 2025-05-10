def swap(i, j, array):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def bubble_sort(array):
    for i in range(len(array)):
        for j in range(len(array) - 1, i, -1):
            if (array[j] < array[j - 1]):
                swap(j - 1, j, array)
        print(array)
    return

if __name__ == "__main__":
    array = [9, 15, 18, 3, 11, 6, 5]
    print("Array before sorted:", array)
    bubble_sort(array)
    print("Array after sorted: ", array)
