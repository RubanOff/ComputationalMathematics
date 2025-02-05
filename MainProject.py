import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


# # Заданные параметры изменяюся интерактивно
# L = 0.1  # Размер грани
# N_x = 100  # Кол-во узлов по оси Х
# N_y = 100  # Кол-во узлов по оси Y
# h_x = L / (N_x - 1)  # Шаг по оси Х
# h_y = L / (N_y - 1)  # Шаг по оси Y
# T_ab = 100  # Температура на грани AB
# T_bc = 200  # Температура на грани BC
# T_cd = 300  # Температура на грани CD
# T_da = 400  # Температура на грани DA
# T_boundary = [T_ab, T_bc, T_cd, T_da]  # Граничные температуры
# max_iterations = 5000  # Количество итераций
# tolerance = 1e-4  # Погрешность

# Создание окна приложения
root = tk.Tk()
root.title("Температурное распределение")

# Создание области для графика
frame_plot = tk.Frame(root)



# Метод Якоби
def jacobi_method(T, T_boundary, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L):
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    N_x = new_N_x
    N_y = new_N_y
    L = new_L
    h_x = L / (N_x - 1)  # Шаг по оси Х
    h_y = L / (N_y - 1)  # Шаг по оси Y

    T_convergence = []
    for _ in range(max_iterations):
        
        # Cоздаем новую копию массива T (на каждой иттерации)
        T_new = np.copy(T)
        for i in range(1, N_y-1):
            for j in range(1, N_x-1):
                # Вычисляем значения внутренних узлов сетки
                T_new[i, j] = ( (h_x**2)*(T[i-1, j] + T[i+1, j]) + (h_y**2)*(T[i, j-1] + T[i, j+1]) ) / (2*(h_x**2 + h_y**2))
        # Вычисляем максимальную разницу между новым распределением температуры T_new и предыдущим распределением T
        max_diff = np.max(np.abs(T_new - T))
        # При превышении температуры выходим из цикла и возвращаем T
        if max_diff < tolerance:
            break
        T_convergence.append(max_diff)
        #
        T = np.copy(T_new)
    return T, T_convergence

# Метод Зейделя с сохранением максимального изменения температуры на каждой итерации
def gauss_seidel_method_with_convergence(T, T_boundary, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L):
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    N_x = new_N_x
    N_y = new_N_y
    L = new_L

    h_x = L / (N_x - 1)  # Шаг по оси Х
    h_y = L / (N_y - 1)  # Шаг по оси Y

    T_convergence = []
    for _ in range(max_iterations):
        max_diff = 0
        for i in range(1, N_y-1):
            for j in range(1, N_x-1):
                T_old = T[i, j]
                # Вычисляем значения внутренних узлов сетки
                T[i, j] = ( (h_x**2)*(T[i-1, j] + T[i+1, j]) + (h_y**2)*(T[i, j-1] + T[i, j+1]) ) / (2*(h_x**2 + h_y**2))
                diff = np.abs(T[i, j] - T_old)
                
                if diff > max_diff:
                    max_diff = diff
        T_convergence.append(max_diff)
        # При превышении температуры выходим из цикла и возвращаем T/Users/eugene/Convergions.png
        if max_diff < tolerance:
            break
    return T, T_convergence

def third_method(T, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L, new_h_t):
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    N_x = new_N_x
    N_y = new_N_y
    L = new_L
    h_x = L / (N_x - 1)  # Шаг по оси Х
    h_y = L / (N_y - 1)  # Шаг по оси Y
    h_t = new_h_t

    T_convergence = []
    for k in range(max_iterations):
        
        # Cоздаем новую копию массива T (на каждой иттерации)
        T = np.vstack([T, [T[k, :, :]]])
        for i in range(1, N_y-1):
            for j in range(1, N_x-1):
                # Вычисляем значения внутренних узлов сетки
                T[k+1, i, j] = T[k, i, j] + h_t * ((T[k, i, j+1] - 2*T[k, i, j] + T[k, i, j-1])/(h_x**2) + (T[k, i+1, j] - 2*T[k, i, j] + T[k, i-1, j])/(h_y**2))
        # Вычисляем максимальную разницу между новым распределением температуры T_new и предыдущим распределением T
        max_diff = np.max(np.abs(T[k+1, :, :] - T[k, :, :]))
        # При превышении температуры выходим из цикла и возвращаем T
        if max_diff < tolerance or max_diff > 1e10:
            break
        T_convergence.append(max_diff)

    return T, T_convergence

# Функция для обновления графиков при изменении параметров
def update_plot():

    # Очистка предыдущих графиков
    for widget in frame_plot.winfo_children():
        widget.destroy()

    new_N_x = entry_N_x.get()
    new_N_y = entry_N_y.get()
    new_T_ab = entry_T_ab.get()
    new_T_bc = entry_T_bc.get()
    new_T_cd = entry_T_cd.get()
    new_T_da = entry_T_da.get()
    new_L = entry_size.get()
    new_h_t = entry_t.get()
    new_max_iterations = entry_iterations.get()
    new_tolerance = entry_error.get()


    # Проверка на пустую строку
    if new_N_x == '' or new_N_y == '' or new_T_ab == '' or new_T_bc == '' or new_T_cd == '' or new_T_da == '' or new_L == '' or new_max_iterations == '' or new_tolerance == '':
        # Вывод сообщения об ошибке
        error_label.config(text="Ошибка: Заполните все поля", fg="red")
        return
    else:
        error_label.config(text=" ", fg="red")
    
    # Преобразование в целые числа и числа с плавающей точкой
    new_N_x = int(new_N_x)
    new_N_y = int(new_N_y)
    new_T_ab = float(new_T_ab)
    new_T_bc = float(new_T_bc)
    new_T_cd = float(new_T_cd)
    new_T_da = float(new_T_da)
    new_L = float(new_L)
    new_h_t = float(new_h_t)
    new_max_iterations = int(new_max_iterations)
    new_tolerance = float(new_tolerance)


    # Вернемся к старым обозначениям
    T_ab = new_T_ab
    T_bc = new_T_bc
    T_cd = new_T_cd
    T_da = new_T_da
    L = new_L
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    T_boundary = [T_ab, T_bc, T_cd, T_da]


    # Построение температурного градиента
    x = np.linspace(0, new_L, new_N_x)
    y = np.linspace(0, new_L, new_N_y)
    X, Y = np.meshgrid(x, y)

    # Задание начальных параметров 
    T = np.zeros((new_N_y, new_N_x))
    T_3 = np.zeros((1, new_N_y, new_N_x))
    for i in range(new_N_y):
        for j in range(new_N_x):
            T[i, 0] = T_ab
            T[new_N_y-1, j] = T_bc
            T[i, new_N_x-1] = T_cd
            T[0, j] = T_da
            T_3[0, i, 0] = T_ab
            T_3[0, new_N_y-1, j] = T_bc
            T_3[0, i, new_N_x-1] = T_cd
            T_3[0, 0, j] = T_da

    # Подсчет температуры во внутренних узлах
    T_solution_jacobi, T_convergence1 = jacobi_method(T, T_boundary, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L)
    T_solution_seidel, T_convergence = gauss_seidel_method_with_convergence(T.copy(), T_boundary, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L)
    T_solution_3, T_convergence_3 = third_method(T_3.copy(), new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L, new_h_t)


    fig, ax = plt.subplots(1, 2)
    c1 = ax[0].contourf(X, Y, T_solution_jacobi, np.linspace(0, 400, 100), cmap='hot')
    ax[0].set_xlabel('x, м')
    ax[0].set_ylabel('y, м')
    ax[0].set_title('Распределение температуры (Якоби)')
    ax[0].text(0, 0, 'A')
    ax[0].text(new_L, 0, 'D', horizontalalignment='right')
    ax[0].text(new_L, new_L, 'C', horizontalalignment='right')
    ax[0].text(0, new_L, 'B')
    c2 = ax[1].contourf(X, Y, T_solution_seidel, np.linspace(0, 400, 100), cmap='hot')
    ax[1].set_xlabel('x, м')
    ax[1].set_ylabel('y, м')
    ax[1].set_title('Распределение температуры (Зейдель)')
    fig.set_size_inches((11,5))
    fig.colorbar(c2)
    fig.colorbar(c1)
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    # canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.get_tk_widget().grid(row=6, rowspan=22, column=1)
    # Важно не перемешить
    canvas.draw()
    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.update()

    print("Рассчет окончен")
    

# Инициализация кнопок
def close_window():
    root.quit()

# def update():
#     # Очистка предыдущих графиков
#     for widget in frame_plot.winfo_children():
#         widget.destroy()

def Third():

    for widget in frame_plot.winfo_children():
        widget.destroy()

    for widget in frame_plot.winfo_children():
        widget.destroy()

    new_N_x = entry_N_x.get()
    new_N_y = entry_N_y.get()
    new_T_ab = entry_T_ab.get()
    new_T_bc = entry_T_bc.get()
    new_T_cd = entry_T_cd.get()
    new_T_da = entry_T_da.get()
    new_L = entry_size.get()
    new_h_t = entry_t.get()
    new_max_iterations = entry_iterations.get()
    new_tolerance = entry_error.get()
    # Проверка на пустую строку
    if new_N_x == '' or new_N_y == '' or new_T_ab == '' or new_T_bc == '' or new_T_cd == '' or new_T_da == '' or new_L == '' or new_max_iterations == '' or new_tolerance == '' or new_h_t == '':
        # Вывод сообщения об ошибке
        error_label.config(text="Ошибка: Заполните все поля", fg="red")
        return
    else:
        error_label.config(text=" ", fg="red")

    # Преобразование в целые числа и числа с плавающей точкой
    new_N_x = int(new_N_x)
    new_N_y = int(new_N_y)
    new_T_ab = float(new_T_ab)
    new_T_bc = float(new_T_bc)
    new_T_cd = float(new_T_cd)
    new_T_da = float(new_T_da)
    new_L = float(new_L)
    new_h_t = float(new_h_t)
    new_max_iterations = int(new_max_iterations)
    new_tolerance = float(new_tolerance)


    # Вернемся к старым обозначениям
    T_ab = new_T_ab
    T_bc = new_T_bc
    T_cd = new_T_cd
    T_da = new_T_da
    L = new_L
    h_t = new_h_t
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    # T_boundary = [T_ab, T_bc, T_cd, T_da]
    
    # Построение температурного градиента
    x = np.linspace(0, new_L, new_N_x)
    y = np.linspace(0, new_L, new_N_y)
    X, Y = np.meshgrid(x, y)

    # Задание начальных параметров 
    T_3 = np.zeros((1, new_N_y, new_N_x))
    for i in range(new_N_y):
        for j in range(new_N_x):
            T_3[0, i, 0] = T_ab
            T_3[0, new_N_y-1, j] = T_bc
            T_3[0, i, new_N_x-1] = T_cd
            T_3[0, 0, j] = T_da

    # Подсчет температуры во внутренних узлах
    T_solution_3, T_convergence = third_method(T_3, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L, new_h_t)


    fig, ax = plt.subplots(1, 2)
    c1 = ax[0].contourf(X, Y, T_solution_3[-1, :, :], np.linspace(0, 400, 100), cmap='hot')
    ax[0].set_xlabel('x, м')
    ax[0].set_ylabel('y, м')
    ax[0].set_title('Распределение температуры (Нестационарное ур-ие)')
    ax[0].text(0, 0, 'A')
    ax[0].text(new_L, 0, 'D', horizontalalignment='right')
    ax[0].text(new_L, new_L, 'C', horizontalalignment='right')
    ax[0].text(0, new_L, 'B')
    plt.plot(range(len(T_convergence)), T_convergence, range(len(T_convergence)), T_convergence)
    ax[1].set_xlabel('Итерация')
    ax[1].set_ylabel('Максимальное изменение температуры')
    ax[1].set_title('Сходимость решения')
    fig.set_size_inches((11,5))
    fig.colorbar(c1)
    

    canvas = FigureCanvasTkAgg(fig, master=root)
    # canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.get_tk_widget().grid(row=0, rowspan=22, column=1)
    # Важно не перемешить
    canvas.draw()
    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.update()

    print("Рассчет окончен")

def Yacoby():
    # Очистка предыдущих графиков
    for widget in frame_plot.winfo_children():
        widget.destroy()

    new_N_x = entry_N_x.get()
    new_N_y = entry_N_y.get()
    new_T_ab = entry_T_ab.get()
    new_T_bc = entry_T_bc.get()
    new_T_cd = entry_T_cd.get()
    new_T_da = entry_T_da.get()
    new_L = entry_size.get()
    new_h_t = entry_t.get()
    new_max_iterations = entry_iterations.get()
    new_tolerance = entry_error.get()
    # Проверка на пустую строку
    if new_N_x == '' or new_N_y == '' or new_T_ab == '' or new_T_bc == '' or new_T_cd == '' or new_T_da == '' or new_L == '' or new_max_iterations == '' or new_tolerance == '' or new_h_t == '':
        # Вывод сообщения об ошибке
        error_label.config(text="Ошибка: Заполните все поля", fg="red")
        return
    else:
        error_label.config(text=" ", fg="red")

    # Преобразование в целые числа и числа с плавающей точкой
    new_N_x = int(new_N_x)
    new_N_y = int(new_N_y)
    new_T_ab = float(new_T_ab)
    new_T_bc = float(new_T_bc)
    new_T_cd = float(new_T_cd)
    new_T_da = float(new_T_da)
    new_L = float(new_L)
    new_max_iterations = int(new_max_iterations)
    new_tolerance = float(new_tolerance)


    # Вернемся к старым обозначениям
    T_ab = new_T_ab
    T_bc = new_T_bc
    T_cd = new_T_cd
    T_da = new_T_da
    L = new_L
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    T_boundary = [T_ab, T_bc, T_cd, T_da]
    
    # Построение температурного градиента
    x = np.linspace(0, new_L, new_N_x)
    y = np.linspace(0, new_L, new_N_y)
    X, Y = np.meshgrid(x, y)

    # Задание начальных параметров 
    T = np.zeros((new_N_y, new_N_x))
    for i in range(new_N_y):
        for j in range(new_N_x):
            T[i, 0] = T_ab
            T[new_N_y-1, j] = T_bc
            T[i, new_N_x-1] = T_cd
            T[0, j] = T_da

    # Подсчет температуры во внутренних узлах
    T_solution_jacobi, T_convergence = jacobi_method(T, T_boundary, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L)


    fig, ax = plt.subplots(1, 2)
    c1 = ax[0].contourf(X, Y, T_solution_jacobi, np.linspace(0, 400, 100), cmap='hot')
    ax[0].set_xlabel('x, м')
    ax[0].set_ylabel('y, м')
    ax[0].set_title('Распределение температуры (Якоби)')
    ax[0].text(0, 0, 'A')
    ax[0].text(new_L, 0, 'D', horizontalalignment='right')
    ax[0].text(new_L, new_L, 'C', horizontalalignment='right')
    ax[0].text(0, new_L, 'B')
    plt.plot(range(len(T_convergence)), T_convergence, range(len(T_convergence)), T_convergence)
    ax[1].set_xlabel('Итерация')
    ax[1].set_ylabel('Максимальное изменение температуры')
    ax[1].set_title('Сходимость метода Якоби')
    fig.set_size_inches((11,5))
    fig.colorbar(c1)
    

    canvas = FigureCanvasTkAgg(fig, master=root)
    # canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.get_tk_widget().grid(row=0, rowspan=22, column=1)
    # Важно не перемешить
    canvas.draw()
    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.update()

    print("Рассчет окончен")
   

def Zeidel():
    # Очистка предыдущих графиков
    for widget in frame_plot.winfo_children():
        widget.destroy()

    new_N_x = entry_N_x.get()
    new_N_y = entry_N_y.get()
    new_T_ab = entry_T_ab.get()
    new_T_bc = entry_T_bc.get()
    new_T_cd = entry_T_cd.get()
    new_T_da = entry_T_da.get()
    new_L = entry_size.get()
    new_h_t = entry_t.get()
    new_max_iterations = entry_iterations.get()
    new_tolerance = entry_error.get()
    # Проверка на пустую строку
    if new_N_x == '' or new_N_y == '' or new_T_ab == '' or new_T_bc == '' or new_T_cd == '' or new_T_da == '' or new_L == '' or new_max_iterations == '' or new_tolerance == '' or new_h_t == '':
        # Вывод сообщения об ошибке
        error_label.config(text="Ошибка: Заполните все поля", fg="red")
        return
    else:
        error_label.config(text=" ", fg="red")

    # Преобразование в целые числа и числа с плавающей точкой
    new_N_x = int(new_N_x)
    new_N_y = int(new_N_y)
    new_T_ab = float(new_T_ab)
    new_T_bc = float(new_T_bc)
    new_T_cd = float(new_T_cd)
    new_T_da = float(new_T_da)
    new_L = float(new_L)
    new_max_iterations = int(new_max_iterations)
    new_tolerance = float(new_tolerance)


    # Вернемся к старым обозначениям
    T_ab = new_T_ab
    T_bc = new_T_bc
    T_cd = new_T_cd
    T_da = new_T_da
    L = new_L
    max_iterations = new_max_iterations
    tolerance = new_tolerance
    T_boundary = [T_ab, T_bc, T_cd, T_da]
    
    # Построение температурного градиента
    x = np.linspace(0, new_L, new_N_x)
    y = np.linspace(0, new_L, new_N_y)
    X, Y = np.meshgrid(x, y)

    # Задание начальных параметров 
    T = np.zeros((new_N_y, new_N_x))
    for i in range(new_N_y):
        for j in range(new_N_x):
            T[i, 0] = T_ab
            T[new_N_y-1, j] = T_bc
            T[i, new_N_x-1] = T_cd
            T[0, j] = T_da

    # Подсчет температуры во внутренних узлах
    T_solution_jacobi, T_convergence = gauss_seidel_method_with_convergence(T, T_boundary, new_max_iterations, new_tolerance, new_N_x, new_N_y, new_L)


    fig, ax = plt.subplots(1, 2)
    c1 = ax[0].contourf(X, Y, T_solution_jacobi, np.linspace(0, 400, 100), cmap='hot')
    ax[0].set_xlabel('x, м')
    ax[0].set_ylabel('y, м')
    ax[0].set_title('Распределение температуры (Зейделя)')
    ax[0].text(0, 0, 'A')
    ax[0].text(new_L, 0, 'D', horizontalalignment='right')
    ax[0].text(new_L, new_L, 'C', horizontalalignment='right')
    ax[0].text(0, new_L, 'B')
    plt.plot(range(len(T_convergence)), T_convergence, range(len(T_convergence)), T_convergence)
    ax[1].set_xlabel('Итерация')
    ax[1].set_ylabel('Максимальное изменение температуры')
    ax[1].set_title('Сходимость метода Зейделя')
    fig.set_size_inches((11,5))
    fig.colorbar(c1)
    

    canvas = FigureCanvasTkAgg(fig, master=root)
    # canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.get_tk_widget().grid(row=0, rowspan=22, column=1)
    # Важно не перемешить
    canvas.draw()
    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.update()

    print("Рассчет окончен")






# Добавление элементов интерфейса

# N_x
message = StringVar()
message.set(100)
label_N_x = tk.Label(root, text="Количество узлов по оси X:")
label_N_x.grid(row=0, column=0)
entry_N_x = tk.Entry(root, textvariable=message)
entry_N_x.grid(row=1, column=0)


# N_y
message = StringVar()
message.set(100)
label_N_y = tk.Label(root, text="Количество узлов по оси Y:")
label_N_y.grid(row=2, column=0)
entry_N_y = tk.Entry(root, textvariable=message)
entry_N_y.grid(row=3, column=0)


# T_ab
message = StringVar()
message.set(100)
label_T_ab = tk.Label(root, text="Температура на грани AB:")
label_T_ab.grid(row=4, column=0)
entry_T_ab = tk.Entry(root, textvariable=message)
entry_T_ab.grid(row=5, column=0)


# T_bc
message = StringVar()
message.set(200)
label_T_bc = tk.Label(root, text="Температура на грани BC:")
label_T_bc.grid(row=6, column=0)
entry_T_bc = tk.Entry(root, textvariable=message)
entry_T_bc.grid(row=7, column=0)


# T_cd
message = StringVar()
message.set(300)
label_T_cd = tk.Label(root, text="Температура на грани CD:")
label_T_cd.grid(row=8, column=0)
entry_T_cd = tk.Entry(root, textvariable=message)
entry_T_cd.grid(row=9, column=0)


# T_da
message = StringVar()
message.set(400)
label_T_da = tk.Label(root, text="Температура на грани DA:")
label_T_da.grid(row=10, column=0)
entry_T_da = tk.Entry(root, textvariable=message)
entry_T_da.grid(row=11, column=0)



# Label size
message = StringVar()
message.set(0.1)
label_size = Label(root, text="Размер грани:")
label_size.grid(row=12, column=0)
entry_size = Entry(root, textvariable=message)
entry_size.grid(row=13, column=0)

message = StringVar()
message.set(1e-7)
label_t = Label(root, text="Шаг по времени:")
label_t.grid(row=14, column=0)
entry_t = Entry(root, textvariable=message)
entry_t.grid(row=15, column=0)


# Max itterations
message = StringVar()
message.set(1000)
label_iterations = Label(root, text="Количество итераций:")
label_iterations.grid(row=16, column=0)
entry_iterations = Entry(root, textvariable=message)
entry_iterations.grid(row=17, column=0)


# Label error
message = StringVar()
message.set(1e-4)
label_error = Label(root, text="Погрешность:")
label_error.grid(row=18, column=0)
entry_error = Entry(root, textvariable=message)
entry_error.grid(row=19, column=0)


# Кнопка для обновления графиков
update_button = tk.Button(root, text="Нестационарное ур-ие", command=Third)
update_button.grid(row=23, column=0)


# Кнопка метода Якоби
jacobi_button = tk.Button(root, text="Якоби", command=Yacoby)
jacobi_button.grid(row=21, column=0)

# Кнопка метода Зейделя
zeidel_button = tk.Button(root, text="Зейдель", command=Zeidel)
zeidel_button.grid(row=22, column=0)

# Кнопка закрытия
close_button = tk.Button(root, text="Закрыть", command=close_window)
close_button.grid(row=24, column=0)

# Текст ошибки если какое-либо значение пустое
error_label = Label(root, text="", fg="red")
error_label.grid(row=25, column=0)

# Запуск основного цикла приложения
root.mainloop()
