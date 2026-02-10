import numpy as np
from PIL import Image
import pandas as pd


# --- Вспомогательные функции для векторной математики ---
def normalize(v):
    """Нормализует вектор."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def dot_product(v1, v2):
    """Вычисляет скалярное произведение двух векторов."""
    return np.dot(v1, v2)


def cross_product(v1, v2):
    """Вычисляет векторное произведение двух векторов."""
    return np.cross(v1, v2)


def clamp(value, min_val, max_val):
    """Ограничивает значение в заданном диапазоне."""
    return max(min_val, min(value, max_val))


# --- Основные формулы освещения ---

# 4. Вычисление вектора нормали плоскости треугольника
def get_normal(P0, P1, P2):
    """
    Вычисляет нормализованный вектор нормали для плоскости треугольника.
    Использует (P1-P0) x (P2-P0) для нормали, смотрящей наружу при CCW обходе.
    """
    normal = cross_product(P1 - P0, P2 - P0)
    return normalize(normal)


# 3. Перевод локальных координат точки в плоскости в глобальные
def get_point_on_triangle(P0, P1, P2, u, v):
    """
    Вычисляет 3D-координаты точки PT на треугольнике
    по барицентрическим координатам u и v (соответствуют x и y в формуле).
    """
    return P0 + (P1 - P0) * u + (P2 - P0) * v


# 1. «Цветная» сила излучения источника под углом к оси источника света
def calculate_light_intensity_I(I0_RGB, O_axis, PL, PT):
    """
    Вычисляет "цветную" силу излучения источника света I(RGB,s).
    s - направление распространения света от PL к PT.
    """
    light_propagation_direction = normalize(PT - PL)

    # cos(theta) - угол между O_axis и направлением распространения света.
    # Ограничиваем до 0, чтобы свет не излучался "назад" относительно оси.
    cos_theta = max(0.0, dot_product(O_axis, light_propagation_direction))

    return I0_RGB * cos_theta


# 2. «Цветная» освещенность точки
def calculate_illuminance_E(PT, PL, I0_RGB, O_axis, N_normalized):
    """
    Вычисляет "цветную" освещенность точки E(RGB,PT).
    """
    # s_vec_to_light - вектор от PT к PL (источнику света)
    s_vec_to_light = PL - PT
    R_squared = dot_product(s_vec_to_light, s_vec_to_light)

    if R_squared < 1e-6:  # Избегаем деления на ноль или очень малые числа
        return np.array([0.0, 0.0, 0.0])

    s_direction_to_light = normalize(s_vec_to_light)

    # I(RGB,s) - сила излучения источника в направлении PT
    I_RGB_at_PT = calculate_light_intensity_I(I0_RGB, O_axis, PL, PT)

    # cos(alpha) - угол между нормалью N и направлением света (от PT к PL).
    # Ограничиваем до 0, чтобы поверхность не освещалась с обратной стороны.
    cos_alpha = max(0.0, dot_product(N_normalized, s_direction_to_light))

    return I_RGB_at_PT * cos_alpha / R_squared


# 8. Средний вектор
def calculate_half_vector_h(V_normalized, s_normalized):
    """
    Вычисляет нормализованный средний вектор h.
    V_normalized - направление наблюдения (от PT к наблюдателю).
    s_normalized - направление света (от PT к источнику света).
    """
    sum_vec = V_normalized + s_normalized
    return normalize(sum_vec)


# 7. Двунаправленная функция отражения (BRDF)
def calculate_brdf_f(K_RGB, kd, ks, ke, V_normalized, s_normalized, N_normalized):
    """
    Вычисляет двунаправленную функцию отражения f(RGB, PT, V, s).
    """
    h_vector = calculate_half_vector_h(V_normalized, s_normalized)

    # dot(h, N) - скалярное произведение h и N, ограничиваем до [0, 1] для зеркального блика.
    h_dot_N = max(0.0, dot_product(h_vector, N_normalized))

    specular_term = ks * (h_dot_N ** ke)

    # K(RGB) - цвет поверхности, модулирующий отраженный свет.
    return K_RGB * (kd + specular_term)


# 6. Яркость точки
def calculate_brightness_L(PT, N_normalized, V_normalized, light_sources, K_RGB, kd, ks, ke):
    """
    Вычисляет яркость точки L(RGB,PT,V) как сумму вкладов от всех источников света.
    """
    total_L_RGB = np.array([0.0, 0.0, 0.0])

    for light in light_sources:
        PL = light['PL']
        I0_RGB = light['I0_RGB']
        O_axis = light['O_axis']

        # Расчет освещенности E для текущего источника света
        E_RGB = calculate_illuminance_E(PT, PL, I0_RGB, O_axis, N_normalized)

        # s_normalized - направление от PT к PL (источнику света)
        s_vec_to_light = PL - PT
        s_normalized = normalize(s_vec_to_light)

        # Расчет BRDF f для текущего источника света
        f_RGB = calculate_brdf_f(K_RGB, kd, ks, ke, V_normalized, s_normalized, N_normalized)

        # L = (1/pi) * E * f
        total_L_RGB += E_RGB * f_RGB

    return (1 / np.pi) * total_L_RGB


# --- Функции для камеры и проекции ---
def look_at_matrix(camera_pos, look_at, up_vec):
    """Создает матрицу вида (View Matrix)."""
    z_axis = normalize(camera_pos - look_at)
    x_axis = normalize(cross_product(up_vec, z_axis))
    y_axis = cross_product(z_axis, x_axis)

    view_matrix = np.identity(4)
    view_matrix[0, :3] = x_axis
    view_matrix[1, :3] = y_axis
    view_matrix[2, :3] = z_axis
    view_matrix[:3, 3] = -np.array([dot_product(x_axis, camera_pos),
                                    dot_product(y_axis, camera_pos),
                                    dot_product(z_axis, camera_pos)])
    return view_matrix


def perspective_matrix(fov_y_deg, aspect_ratio, near, far):
    """Создает матрицу перспективной проекции (Projection Matrix)."""
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    proj_matrix = np.zeros((4, 4))
    proj_matrix[0, 0] = f / aspect_ratio
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    return proj_matrix


def transform_point(point, matrix):
    """Преобразует 3D-точку с помощью матрицы (включая перспективное деление)."""
    p_homogeneous = np.append(point, 1.0)
    p_transformed = np.dot(matrix, p_homogeneous)
    return p_transformed[:3] / p_transformed[3]


def map_to_screen(point_proj, width, height):
    """
    Преобразует точку из NDC-пространства [-1, 1] в экранные координаты.
    Y-ось инвертирована для соответствия стандартным экранным координатам.
    """
    screen_x = int((point_proj[0] + 1.0) * 0.5 * width)
    screen_y = int((1.0 - (point_proj[1] + 1.0) * 0.5) * height)
    return np.array([screen_x, screen_y, point_proj[2]])  # Сохраняем Z для потенциального использования


# Барицентрические координаты для 2D-точки p в треугольнике (a, b, c)
def barycentric_coords_2d(p, a, b, c):
    """
    Вычисляет барицентрические координаты (u, v, w) 2D-точки p
    относительно 2D-треугольника (a, b, c).
    Возвращает (u, v, w) где p = u*a + v*b + w*c.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = dot_product(v0, v0)
    d01 = dot_product(v0, v1)
    d11 = dot_product(v1, v1)
    d20 = dot_product(v2, v0)
    d21 = dot_product(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:  # Вырожденный треугольник
        return -1, -1, -1

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


if __name__ == "__main__":
    # --- Параметры сцены ---
    # Вершины треугольника
    P0 = np.array([-1.0, -1.0, 0.0])
    P1 = np.array([1.0, -1.0, 0.0])
    P2 = np.array([0.0, 1.0, 0.0])

    # Источники света
    LIGHT_SOURCES = [
        {
            'PL': np.array([-2.0, 2.0, 2.0]),  # Положение источника 1
            'I0_RGB': np.array([500.0, 0.0, 0.0]),  # Красный свет
            'O_axis': normalize(np.array([1.0, -1.0, -1.0]))  # Ось излучения (направлена к центру)
        },
        {
            'PL': np.array([2.0, 2.0, 2.0]),  # Положение источника 2
            'I0_RGB': np.array([0.0, 0.0, 500.0]),  # Синий свет
            'O_axis': normalize(np.array([-1.0, -1.0, -1.0]))  # Ось излучения (направлена к центру)
        }
    ]

    # Свойства поверхности
    K_RGB = np.array([0.8, 0.8, 0.8])  # Цвет поверхности (светлый серый, масштабированный 0-1)
    KD = 0.7  # Коэффициент диффузного отражения
    KS = 0.3  # Коэффициент зеркального отражения
    KE = 10.0  # Показатель степени для зеркального блика (глянцевость)

    # --- Параметры изображения и камеры ---
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720
    OUTPUT_FILENAME = "triangle.png"

    CAMERA_POSITION = np.array([0.0, 0.0, 5.0])  # Положение камеры
    LOOK_AT = np.array([0.0, 0.0, 0.0])  # Точка, на которую смотрит камера
    UP_VECTOR = np.array([0.0, 1.0, 0.0])  # Вектор "вверх" для камеры
    FOV_Y_DEG = 60.0  # Угол обзора по вертикали в градусах
    ASPECT_RATIO = IMAGE_WIDTH / IMAGE_HEIGHT
    NEAR_PLANE = 0.1  # Ближняя плоскость отсечения
    FAR_PLANE = 100.0  # Дальняя плоскость отсечения

    # --- Предварительные расчеты ---
    # Нормаль к плоскости треугольника (одна для всего треугольника)
    N_triangle_normalized = get_normal(P0, P1, P2)

    # --- Генерация изображения ---
    img_array = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                         dtype=np.float32)  # Используем float32 для промежуточных расчетов яркости
    max_brightness_overall = 0.0  # Для последующего масштабирования

    # 1. Настройка матриц камеры
    view_matrix = look_at_matrix(CAMERA_POSITION, LOOK_AT, UP_VECTOR)
    proj_matrix = perspective_matrix(FOV_Y_DEG, ASPECT_RATIO, NEAR_PLANE, FAR_PLANE)

    # 2. Преобразование вершин треугольника в экранные координаты
    P0_view = transform_point(P0, view_matrix)
    P1_view = transform_point(P1, view_matrix)
    P2_view = transform_point(P2, view_matrix)

    P0_proj = transform_point(P0_view, proj_matrix)
    P1_proj = transform_point(P1_view, proj_matrix)
    P2_proj = transform_point(P2_view, proj_matrix)

    P0_screen = map_to_screen(P0_proj, IMAGE_WIDTH, IMAGE_HEIGHT)
    P1_screen = map_to_screen(P1_proj, IMAGE_WIDTH, IMAGE_HEIGHT)
    P2_screen = map_to_screen(P2_proj, IMAGE_WIDTH, IMAGE_HEIGHT)

    # Извлечение 2D-координат для расчета барицентрических координат
    P0_2d = P0_screen[:2]
    P1_2d = P1_screen[:2]
    P2_2d = P2_screen[:2]

    # 3. Растеризация
    # Определение ограничивающего прямоугольника проекции треугольника
    min_x = int(min(P0_screen[0], P1_screen[0], P2_screen[0]))
    max_x = int(max(P0_screen[0], P1_screen[0], P2_screen[0]))
    min_y = int(min(P0_screen[1], P1_screen[1], P2_screen[1]))
    max_y = int(max(P0_screen[1], P1_screen[1], P2_screen[1]))

    # Ограничение прямоугольника размерами изображения
    min_x = max(0, min_x)
    max_x = min(IMAGE_WIDTH - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(IMAGE_HEIGHT - 1, max_y)

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            p_2d = np.array([px, py])

            # Расчет 2D барицентрических координат
            b0, b1, b2 = barycentric_coords_2d(p_2d, P0_2d, P1_2d, P2_2d)

            # Проверка, находится ли пиксель внутри треугольника (с небольшим допуском)
            epsilon = 1e-3
            if b0 >= -epsilon and b1 >= -epsilon and b2 >= -epsilon:
                # Восстановление 3D-точки PT с использованием барицентрических координат
                PT_3D = b0 * P0 + b1 * P1 + b2 * P2

                # Расчет направления наблюдения для текущей точки (от PT к камере)
                V_normalized_per_pixel = normalize(CAMERA_POSITION - PT_3D)

                # Расчет яркости для этой точки
                L_RGB_float = calculate_brightness_L(PT_3D, N_triangle_normalized, V_normalized_per_pixel,
                                                     LIGHT_SOURCES, K_RGB, KD, KS, KE)

                # Сохраняем как float32
                img_array[py, px] = L_RGB_float

                # Обновляем максимальное значение яркости для последующего масштабирования
                max_brightness_overall = max(max_brightness_overall, np.max(L_RGB_float))

    # --- Масштабирование ---
    if max_brightness_overall > 0:
        # Масштабируем яркость так, чтобы максимальное значение стало 1.0 (или чуть меньше)
        scale_factor = 1.0 / max_brightness_overall
        img_array_scaled = img_array * scale_factor

        # Ограничиваем значения до 0-255 и преобразуем в uint8
        final_img_array = np.clip(img_array_scaled * 255.0, 0, 255).astype(np.uint8)
    else:
        # Если все значения яркости нулевые, просто создаем черное изображение
        final_img_array = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # Сохранение изображения
    img = Image.fromarray(final_img_array, 'RGB')
    img.save(OUTPUT_FILENAME)
    print(f"Изображение сохранено как {OUTPUT_FILENAME}")

    # --- Генерация таблиц ---

    # Определяем сетку для u, v (локальные координаты x, y в задании)
    num_samples = 5
    u_coords = np.linspace(0, 1, num_samples)
    v_coords = np.linspace(0, 1, num_samples)

    columns = [f"x={u:.2f}" for u in u_coords]
    index = [f"y={v:.2f}" for v in v_coords]

    print("\n--- Вычисленные значения освещенности E (RGB, PT) с учетом обоих источников, "
          "для точек, заданных локальными координатами ---")
    illuminance_E_total_data = []
    for u_val in u_coords:
        row_data = []
        for v_val in v_coords:
            if u_val + v_val <= 1.0 + 1e-6:
                PT_sample = get_point_on_triangle(P0, P1, P2, u_val, v_val)
                E1_RGB = calculate_illuminance_E(PT_sample, LIGHT_SOURCES[0]['PL'], LIGHT_SOURCES[0]['I0_RGB'],
                                                 LIGHT_SOURCES[0]['O_axis'], N_triangle_normalized)
                E2_RGB = calculate_illuminance_E(PT_sample, LIGHT_SOURCES[1]['PL'], LIGHT_SOURCES[1]['I0_RGB'],
                                                 LIGHT_SOURCES[1]['O_axis'], N_triangle_normalized)
                E_total_RGB = E1_RGB + E2_RGB
                row_data.append(f"({int(E_total_RGB[0])}, {int(E_total_RGB[1])}, {int(E_total_RGB[2])})")
            else:
                row_data.append("       -       ")
        illuminance_E_total_data.append(row_data)

    df_E_total = pd.DataFrame(illuminance_E_total_data, index=index, columns=columns)
    print(df_E_total.to_string())

    print("\n--- Соответствие локальных координат (x, y) глобальным (X, Y, Z) ---")
    coords_correspondence = []
    for u_val in u_coords:
        for v_val in v_coords:
            if u_val + v_val <= 1.0 + 1e-6:
                PT_sample = get_point_on_triangle(P0, P1, P2, u_val, v_val)
                coords_correspondence.append((u_val, v_val, PT_sample))

    for (u, v, pt) in coords_correspondence:
        print(f"Локальные (x={u:.2f}, y={v:.2f}) -> Глобальные ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")

    print("\n--- Таблица освещенности E (RGB, PT) по глобальным координатам ---")

    global_x_coords = np.arange(-1, 1.01, 0.25)
    global_y_coords = np.arange(-1, 1.01, 0.25)

    columns = [f"X={x:.2f}" for x in global_x_coords]
    index = [f"Y={y:.2f}" for y in global_y_coords]

    illuminance_global_data = []
    for y_val in global_y_coords:
        row_data = []
        for x_val in global_x_coords:
            point_found = False
            for (u_local, v_local, pt_global) in coords_correspondence:
                if abs(pt_global[0] - x_val) < 1e-6 and abs(pt_global[1] - y_val) < 1e-6 and abs(pt_global[2]) < 1e-6:
                    E1_RGB = calculate_illuminance_E(pt_global, LIGHT_SOURCES[0]['PL'], LIGHT_SOURCES[0]['I0_RGB'],
                                                     LIGHT_SOURCES[0]['O_axis'], N_triangle_normalized)
                    E2_RGB = calculate_illuminance_E(pt_global, LIGHT_SOURCES[1]['PL'], LIGHT_SOURCES[1]['I0_RGB'],
                                                     LIGHT_SOURCES[1]['O_axis'], N_triangle_normalized)
                    E_total_RGB = E1_RGB + E2_RGB
                    row_data.append(f"({int(E_total_RGB[0])}, {int(E_total_RGB[1])}, {int(E_total_RGB[2])})")
                    point_found = True
                    break

            if not point_found:
                row_data.append("       -       ")
        illuminance_global_data.append(row_data)

    df_E_global = pd.DataFrame(illuminance_global_data, index=index, columns=columns)
    print(df_E_global.to_string())

    print("\n--- Вычисленные значения яркостей L (RGB, PT, V) с учетом обоих источников ---")
    num_samples = 5
    u_coords = np.linspace(0, 1, num_samples)
    v_coords = np.linspace(0, 1, num_samples)

    columns_l = [f"x={u:.2f}" for u in u_coords]
    index_l = [f"y={v:.2f}" for v in v_coords]

    brightness_L_data = []
    for v_val in v_coords:
        row_data = []
        for u_val in u_coords:
            if u_val + v_val <= 1.0 + 1e-6:
                PT_sample = get_point_on_triangle(P0, P1, P2, u_val, v_val)
                V_normalized_per_sample = normalize(CAMERA_POSITION - PT_sample)
                L_RGB_float = calculate_brightness_L(PT_sample, N_triangle_normalized, V_normalized_per_sample,
                                                     LIGHT_SOURCES, K_RGB, KD, KS, KE)
                row_data.append(f"({L_RGB_float[0]:.2f}, {L_RGB_float[1]:.2f}, {L_RGB_float[2]:.2f})")
            else:
                row_data.append("       -       ")
        brightness_L_data.append(row_data)

    df_L = pd.DataFrame(brightness_L_data, index=index_l, columns=columns_l)
    print(df_L.to_string())