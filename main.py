import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    N = int(input("Введите количество строк (столбцов) квадратной матрицы больше 3 и меньше 184:"))
    if (N >= 4) and (N <= 183):
        K = int(input("Введите число К:"))
        program = time.time()
        start = time.time()
        A = np.zeros((N, N), dtype=int)
        F = np.zeros((N, N), dtype=int)
        for i in range(N):                                         #формируем матрицу А
            for j in range(N):
                A[i][j] = np.random.randint(-10, 10)
        middle = time.time()
        print("Матрица A:\n", A, "\nВремя:", middle - start)
        for i in range(N):                                         #формируем матрицу F копируя из матрицы А
            for j in range(N):
                F[i][j] = A[i][j]
        n = N // 2                                                 #размерность подматрицы
        start = time.time()
        С = np.zeros((n, n), dtype=int)                            #формируем матрицу C
        for i in range(n):
            for j in range(n):
                С[i][j] = A[N - i - 1][N - j - 1]
        middle = time.time()
        print("Матрица С:\n", С, "\nВремя:", middle - start)
        prost = 0
        nul = 0
        for i in range(n):
            for j in range(n):
                if j % 2 == 1 and С[i][j] % 2 == 1:                #количество простых чисел в нечетных столбцах
                    prost += 1
                if i % 2 == 0 and С[i][j] == 0:                    #кол во 0 элементов в четных строках
                    nul += 1
        print("Количество простых чисел в нечётных столбцах:", prost, "\nкол во нулей чётных строках:", nul)
        if prost > nul:
            print("\nМеняем E и C симметрично")
            for i in range(n):                                      #E и C симметрично
                for j in range(n):
                    F[N - i - 1][N - j - 1] = A[i][j]
                    F[i][j] = A[N - i - 1][N - j - 1]
        else:
            print("\nМеняем C и B несимметрично")
            for i in range(n):                                      #C и B несимметрично
                for j in range(n):
                    F[i+n][j+n] = A[i][n + j]
                    F[i][j+n] = A[i+n][j+n]
        print("Матрица A:\n", A, "\nМатрица F:\n", F)
        print("Определитель матрицы А:", round(np.linalg.det(A)), "\nСумма диагональных элементов матрицы F:", np.trace(F))
        G = np.tril(A, k=0)
        if np.linalg.det(A) == 0 or np.linalg.det(F) == 0:
            print("Нельзя вычислить т.к. матрица A или F вырождена")
        elif np.linalg.det(A) > np.trace(F):
            print("Вычисление выражения: A^-1*A^T-K*F^-1")
            A = np.dot(np.linalg.inv(A), np.transpose(A)) - np.dot(F, K)        #A^-1*A^T – K * F
        else:
            print("Вычисление выражения: (A^-1+G-F^-1)*K")
            A = (np.transpose(A) + np.linalg.inv(G) - np.linalg.inv(F)) * K     #(A^Т +G^-1-F^-1)*K
        print("Результат:")
        for i in A:                                                             #Вывод результата
            for j in i:
                print("%5d" % round(j), end=' ')
            print()
        finish = time.time()
        result = finish - program
        print("Время работы программы: " + str(result) + " секунды.")
        plt.title("Значения элементов матрицы")  # 1 пример matplotlib
        plt.xlabel("Номер числа в строке")
        plt.ylabel("Значение элемента")
        for j in range(N):
            plt.plot([i for i in range(N)], A[j][::], marker='o')
        plt.show()
        plt.matshow(A)  # 2 пример matplotlib
        plt.show()
        fig = plt.figure()  # 3 пример matplotlib
        ax1 = fig.add_subplot(121)
        ax1.imshow(A, interpolation='bilinear', cmap=cm.Greys_r)
        ax2 = fig.add_subplot(122)
        ax2.imshow(A, interpolation='nearest', cmap=cm.Greys_r)
        plt.show()
        sns.set_theme()  # 1 пример seaborn
        uniform_data = A
        if N >= 15 or K >= 10:
            heatmap = sns.heatmap(A, vmin=-5 * N, vmax=5 * N)
        elif N < 15 and K < 10:
            heatmap = sns.heatmap(A, vmin=-20, vmax=20, annot_kws={'size': 7}, annot=True, fmt=".1f")
        plt.show()
        sns.set_theme(style="darkgrid")  # 2 пример seaborn
        df = pd.DataFrame(A)
        p = sns.lineplot(data=df)
        p.set_xlabel("Номер элемента в столбце", fontsize=12)
        p.set_ylabel("Значение", fontsize=12)
        sns.relplot(sort=False, kind="line", data=df)
        plt.show()
        sns.catplot(data=df, kind="box")  # 3 пример seaborn
        plt.show()
    else:
        print("\nВы ввели неверное число.")
except ValueError:
    print("\nЭто не число")