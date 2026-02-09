import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

# Вершины треугольника (3D)
P0 = np.array([-1.0, 0.0, 0.0])
P1 = np.array([1.0, 0.0, 0.0])
P2 = np.array([0.0, 2.0, 0.5])

# Источники света
lights = [
    {
        'I0': np.array([1.0, 0.8, 0.7]),  # RGB интенсивность
        'O': np.array([0.0, 0.0, -1.0]),  # Направление оси
        'PL': np.array([0.0, 0.0, 3.0])  # Положение источника
    },
    {
        'I0': np.array([0.6, 0.8, 1.0]),
        'O': np.array([-0.5, -0.5, -1.0]),
        'PL': np.array([-2.0, -1.0, 2.0])
    }
]

# Параметры поверхности
K_RGB = np.array([0.9, 0.7, 0.8])  # Цвет поверхности
k_d = 0.8  # Коэффициент диффузного отражения
k_s = 0.5  # Коэффициент зеркального отражения
k_e = 10.0  # Коэффициент ширины блика

# Направление наблюдения
V = np.array([0.0, 0.0, 1.0])

# Локальные координаты точек для таблиц (5x5 точек)
x_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
y_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

# Параметры изображения
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def normalize(v):
    """Нормализация вектора"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def local_to_global(P0, P1, P2, x, y):
    """Перевод локальных координат в глобальные (формула 3)"""
    e1 = normalize(P1 - P0)
    e2 = normalize(P2 - P0)
    return P0 + e1 * x + e2 * y


def triangle_normal(P0, P1, P2):
    """Вычисление нормали треугольника (формула 4)"""
    v1 = P1 - P0
    v2 = P2 - P0
    normal = np.cross(v2, v1)  # Обратный порядок для правильной ориентации
    return normalize(normal)


def calculate_illumination(P_T, light, N):
    """Расчет освещенности точки от одного источника"""
    # Вектор от точки до источника
    s = light['PL'] - P_T
    R2 = np.dot(s, s)

    if R2 == 0:
        return np.zeros(3)

    s_norm = normalize(s)

    # Угол между направлением света и осью источника (формула 1)
    cos_theta = max(0, np.dot(s_norm, normalize(light['O'])))
    I_RGB = light['I0'] * cos_theta

    # Угол между направлением света и нормалью
    cos_alpha = max(0, np.dot(s_norm, N))

    # Освещенность (формула 2)
    E_RGB = I_RGB * cos_alpha / R2
    return E_RGB


def calculate_brdf(P_T, V, s, N, K_RGB, k_d, k_s, k_e):
    """Двунаправленная функция отражения (формула 7)"""
    h = normalize(V + s)  # Средний вектор (формула 8)
    specular = k_s * (max(0, np.dot(h, N)) ** k_e)
    return K_RGB * (k_d + specular)


def calculate_brightness(P_T, V, lights, N, K_RGB, k_d, k_s, k_e):
    """Расчет яркости точки (формула 6)"""
    L_RGB = np.zeros(3)

    for light in lights:
        s = light['PL'] - P_T
        s_norm = normalize(s)

        # Освещенность от источника
        E_RGB = calculate_illumination(P_T, light, N)

        # BRDF
        f_RGB = calculate_brdf(P_T, V, s_norm, N, K_RGB, k_d, k_s, k_e)

        # Вклад источника
        L_RGB += E_RGB * f_RGB

    return L_RGB / np.pi


def barycentric_coords(P, A, B, C):
    """Вычисление барицентрических координат точки P относительно треугольника ABC"""
    v0 = B - A
    v1 = C - A
    v2 = P - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


def is_point_in_triangle(P, A, B, C):
    """Проверка, лежит ли точка внутри треугольника"""
    coords = barycentric_coords(P, A, B, C)
    if coords is None:
        return False
    u, v, w = coords
    return (u >= 0) and (v >= 0) and (w >= 0) and (u <= 1) and (v <= 1) and (w <= 1)


# Нормаль треугольника
N = triangle_normal(P0, P1, P2)

# Таблицы для вывода
table_points = []
E1_table = []
E2_table = []
L_table = []

print("=" * 80)
print("ТАБЛИЦЫ ОСВЕЩЕННОСТИ И ЯРКОСТИ")
print("=" * 80)

# Создаем сетку точек для таблиц
for y in y_vals:
    for x in x_vals:
        # Проверяем, что точка внутри треугольника в локальных координатах
        if x + y <= 1.0:  # Упрощенная проверка
            P_T = local_to_global(P0, P1, P2, x, y)

            # Освещенность от первого источника
            E1 = calculate_illumination(P_T, lights[0], N)

            # Освещенность от второго источника
            E2 = calculate_illumination(P_T, lights[1], N)

            # Яркость
            L = calculate_brightness(P_T, V, lights, N, K_RGB, k_d, k_s, k_e)

            # Сохраняем для таблицы
            table_points.append((x, y))
            E1_table.append(E1)
            E2_table.append(E2)
            L_table.append(L)

# Вывод таблицы E1
print("\n" + "=" * 80)
print("Таблица 1: Освещенность от первого источника E1(RGB, P_T)")
print("=" * 80)
print(f"{'x':>8} {'y':>8} {'R':>12} {'G':>12} {'B':>12}")
print("-" * 60)

for i, (x, y) in enumerate(table_points):
    E1 = E1_table[i]
    print(f"{x:8.3f} {y:8.3f} {E1[0]:12.6f} {E1[1]:12.6f} {E1[2]:12.6f}")

# Вывод таблицы E2
print("\n" + "=" * 80)
print("Таблица 2: Освещенность от второго источника E2(RGB, P_T)")
print("=" * 80)
print(f"{'x':>8} {'y':>8} {'R':>12} {'G':>12} {'B':>12}")
print("-" * 60)

for i, (x, y) in enumerate(table_points):
    E2 = E2_table[i]
    print(f"{x:8.3f} {y:8.3f} {E2[0]:12.6f} {E2[1]:12.6f} {E2[2]:12.6f}")

# Вывод таблицы яркостей
print("\n" + "=" * 80)
print("Таблица 3: Яркость L(RGB, P_T, V)")
print("=" * 80)
print(f"{'x':>8} {'y':>8} {'R':>12} {'G':>12} {'B':>12}")
print("-" * 60)

for i, (x, y) in enumerate(table_points):
    L = L_table[i]
    print(f"{x:8.3f} {y:8.3f} {L[0]:12.6f} {L[1]:12.6f} {L[2]:12.6f}")

print("\n" + "=" * 80)
print("Создание изображения треугольника...")
print("=" * 80)

# Создаем фигуру
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.axis('off')

# Проекция вершин на 2D (X,Y)
tri_2d = np.array([P0[:2], P1[:2], P2[:2]])

# Создаем сетку для растеризации
x_grid = np.linspace(-2, 2, IMAGE_WIDTH)
y_grid = np.linspace(-1, 3, IMAGE_HEIGHT)

# Создаем пустое изображение
image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

# Заполняем треугольник
for i, y in enumerate(y_grid):
    for j, x in enumerate(x_grid):
        P_test = np.array([x, y, 0])
        A_2d = np.append(tri_2d[0], 0)
        B_2d = np.append(tri_2d[1], 0)
        C_2d = np.append(tri_2d[2], 0)

        if is_point_in_triangle(P_test, A_2d, B_2d, C_2d):
            # Находим барицентрические координаты
            u, v, w = barycentric_coords(P_test, A_2d, B_2d, C_2d)

            # Восстанавливаем 3D координаты
            P_3d = u * P0 + v * P1 + w * P2

            # Вычисляем яркость
            brightness = calculate_brightness(P_3d, V, lights, N, K_RGB, k_d, k_s, k_e)

            # Ограничиваем значения и устанавливаем цвет
            color = np.clip(brightness, 0, 1)
            image[i, j] = color

# Отображаем изображение
ax.imshow(image, extent=[-2, 2, -1, 3], origin='lower')

# Добавляем контур треугольника
triangle = Polygon(tri_2d, closed=True, fill=False,
                   edgecolor='white', linewidth=2)
ax.add_patch(triangle)

# Сохраняем изображение
output_filename = "illuminated_triangle.png"
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Изображение сохранено как: {output_filename}")
print("Размер изображения: 1280x720 пикселей")
print("=" * 80)
print("Программа завершена успешно!")