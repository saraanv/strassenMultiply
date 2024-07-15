import time
import numpy as np
import matplotlib.pyplot as plt

class MatrixMultiplier:

    # constructor
    def __init__(self):
        pass

    # making a random matrix
    def make_random_matrix(self, size):
        return np.random.randint(0, 10, size=(size, size))


    #calculating the matrices multiply in normal way
    def normal_matrix_multiply(self, A, B):
        n = len(A)
        C = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]

        return C


    # calculating matrices multiply in D&C way
    def divide_and_conquer_multiply(self, A, B):
        def divide_and_conquer(A, B):
            n = len(A)

            if n == 1:
                return np.array([[A[0][0] * B[0][0]]])

            mid = n // 2

            A11 = A[:mid, :mid]
            A12 = A[:mid, mid:]
            A21 = A[mid:, :mid]
            A22 = A[mid:, mid:]

            B11 = B[:mid, :mid]
            B12 = B[:mid, mid:]
            B21 = B[mid:, :mid]
            B22 = B[mid:, mid:]

            C11 = divide_and_conquer(A11, B11) + divide_and_conquer(A12, B21)
            C12 = divide_and_conquer(A11, B12) + divide_and_conquer(A12, B22)
            C21 = divide_and_conquer(A21, B11) + divide_and_conquer(A22, B21)
            C22 = divide_and_conquer(A21, B12) + divide_and_conquer(A22, B22)

            C = np.zeros((n, n))
            C[:mid, :mid] = C11
            C[:mid, mid:] = C12
            C[mid:, :mid] = C21
            C[mid:, mid:] = C22

            return C

        size = len(A)
        if size == len(B):
            return divide_and_conquer(A, B)
        else:
            raise ValueError("Matrix size don't match")


    # calculating the matrices multiply in strassen way
    def strassen_matrix_multiply(self, A, B):
        def strassen(A, B):
            n = A.shape[0]

            if n == 1:
                return A * B

            mid = n // 2
            A11 = A[:mid, :mid]
            A12 = A[:mid, mid:]
            A21 = A[mid:, :mid]
            A22 = A[mid:, mid:]

            B11 = B[:mid, :mid]
            B12 = B[:mid, mid:]
            B21 = B[mid:, :mid]
            B22 = B[mid:, mid:]

            M1 = strassen(A11 + A22, B11 + B22)
            M2 = strassen(A21 + A22, B11)
            M3 = strassen(A11, B12 - B22)
            M4 = strassen(A22, B21 - B11)
            M5 = strassen(A11 + A12, B22)
            M6 = strassen(A21 - A11, B11 + B12)
            M7 = strassen(A12 - A22, B21 + B22)

            C11 = M1 + M4 - M5 + M7
            C12 = M3 + M5
            C21 = M2 + M4
            C22 = M1 - M2 + M3 + M6

            C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

            return C

        n = A.shape[0]

        if n == B.shape[0] and (n & (n - 1)) == 0:
            return strassen(A, B)
        else:
            raise ValueError("Matrix size is not a power 2 / Matrices are not square")


    # calculating the time using system time
    def execute_time(self, func, *args):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        return end_time - start_time

    def compare_multiply(self, n):
        normal_times = []
        divide_and_conquer_times = []
        strassen_times = []

        sizes = [2**i for i in range(1, n+1)]

        for size in sizes:
            A = self.make_random_matrix(size)
            B = self.make_random_matrix(size)

            normal_time = self.execute_time(self.normal_matrix_multiply, A, B)
            normal_times.append(normal_time)

            divide_and_conquer_time = self.execute_time(self.divide_and_conquer_multiply, A, B)
            divide_and_conquer_times.append(divide_and_conquer_time)

            strassen_time = self.execute_time(self.strassen_matrix_multiply, A, B)
            strassen_times.append(strassen_time)

        return sizes, normal_times, divide_and_conquer_times, strassen_times


    # making a plot to compare the multiply methods
    def plot_compare(self, sizes, normal_times, divide_and_conquer_times, strassen_times):
        plt.figure(figsize=(8, 5))
        plt.plot(sizes, normal_times, marker='o', linestyle='-', color='b', label='Normal')
        plt.plot(sizes, divide_and_conquer_times, marker='s', linestyle='--', color='g', label='Divide & Conquer')
        plt.plot(sizes, strassen_times, marker='^', linestyle='-.', color='r', label='Strassen\'s')
        plt.title(f'Comparing Matrix Multiplication Methods')
        plt.xlabel('Matrix Size')
        plt.ylabel('Execute Time (s)')
        plt.xticks(sizes)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Numeric Results:")
        print("Matrix Size\tNormal Time (s)\tDivide & Conquer Time (s)\tStrassen Time (s)")
        for i, size in enumerate(sizes):
            print(f"{size}\t\t{normal_times[i]:.6f}\t\t\t{divide_and_conquer_times[i]:.6f}\t\t\t{strassen_times[i]:.6f}")



# taking a matrix from user
def input_matrix(prompt):
    print(prompt)
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    matrix = []
    print("Enter the matrix values in row:")
    for i in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)
    return np.array(matrix)


# main program
if __name__ == "__main__":
    multiplier = MatrixMultiplier()

    while True:
        try:
            n = int(input("Enter the int for power of 2 (for size 2^n): "))
            if n <= 0:
                print("Please enter a positive integer greater than 0.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    sizes, normal_times, divide_and_conquer_times, strassen_times = multiplier.compare_multiply(n)

    multiplier.plot_compare(sizes, normal_times, divide_and_conquer_times, strassen_times)

    while True:
        print("\nEnter matrices A and B")
        A = input_matrix("Matrix A:")
        B = input_matrix("Matrix B:")

        if A.shape != B.shape:
            print("Error: Matrices must be of the same size. Please try again.")
        else:
            try:
                C = multiplier.strassen_matrix_multiply(A, B)

                print("\nMatrix A:")
                print(A)
                print("\nMatrix B:")
                print(B)
                print("\nMatrix C (A * B) Strassen:")
                print(C)

                break
            except ValueError as e:
                print(e)
