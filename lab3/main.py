import matplotlib.pyplot as plt
import numpy as np

# Флаг для управления отрисовкой графиков
# True - графики отображаются, False - только вывод в консоль
SHOW_PLOTS = True

rng = np.random.default_rng(2)


def generate_random_triangle_point(p0, p1, p2):
    u, v = p1 - p0, p2 - p0
    e_u, e_v = rng.uniform(), rng.uniform()

    if e_u + e_v > 1:
        e_u = 1 - e_u
        e_v = 1 - e_v

    return p0 + u * e_u + v * e_v


def generate_random_circle_point(n, r, c):
    n = n / np.linalg.norm(n)

    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    while True:
        e_u, e_v = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        if e_u ** 2 + e_v ** 2 <= 1.0:
            break

    return c + r * (u * e_u + v * e_v)


def generate_random_sphere_point(r, c):
    e_phi, e_theta = rng.uniform(), rng.uniform()

    phi = 2 * np.pi * e_phi
    theta = np.acos(2 * e_theta - 1)

    return c + np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])


def generate_random_cosine_direction(n):
    n = n / np.linalg.norm(n)

    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    e_phi, e_theta = rng.uniform(), rng.uniform()
    phi = 2.0 * np.pi * e_phi
    sin_theta = np.sqrt(e_theta)
    cos_theta = np.sqrt(1.0 - e_theta)

    d = u * (np.cos(phi) * sin_theta) + v * (np.sin(phi) * sin_theta) + n * cos_theta
    return d / np.linalg.norm(d)


def point_in_triangle(point, p0, p1, p2):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, p0, p1)
    d2 = sign(point, p1, p2)
    d3 = sign(point, p2, p0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def point_in_square(point, square_center, half_size):
    x_min = square_center[0] - half_size
    x_max = square_center[0] + half_size
    y_min = square_center[1] - half_size
    y_max = square_center[1] + half_size
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def point_in_spherical_circle(point, center, radius, axis, cos_theta_max):
    vec = point - center
    r = np.linalg.norm(vec)
    if abs(r - radius) > 0.01:
        return False
    direction = vec / r
    cos_theta = np.dot(direction, axis)
    return cos_theta >= cos_theta_max


def plot_triangle_points(vertices, n_points):
    p0, p1, p2 = vertices
    triangle_points = np.array([
        generate_random_triangle_point(p0, p1, p2)
        for _ in range(n_points)
    ])

    # Выбираем 3 квадрата, гарантированно лежащие внутри треугольника
    square1_center = np.array([1.5, 1.6])
    square2_center = np.array([2.8, 1.8])
    square3_center = np.array([3.6, 1])

    half_size = 0.4
    squares = [square1_center, square2_center, square3_center]
    colors = ['red', 'orange', 'darkred']

    # Подсчет точек в каждом квадрате
    counts = []
    for center in squares:
        count = sum(1 for p in triangle_points if point_in_square(p, center, half_size))
        counts.append(count)

    mean_count = np.mean(counts)
    deviations = [abs(c - mean_count) / mean_count * 100 for c in counts]
    max_deviation = max(deviations)

    print(f"\n=== Треугольник ===")
    print(f"Площадь каждого квадрата: {(2 * half_size) ** 2:.2f}")
    for i, count in enumerate(counts, 1):
        print(f"Количество точек в квадрате {i} (центр {squares[i - 1]}): {count}")
    print(f"Среднее количество: {mean_count:.1f}")
    print(f"Максимальное отклонение от среднего: {max_deviation:.2f}%")

    if not SHOW_PLOTS:
        print("(График пропущен: SHOW_PLOTS = False)")
        return

    triangle_vertices = np.vstack([vertices, vertices[0]])

    plt.figure(figsize=(7, 6))
    plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color="black", linewidth=1.5)
    plt.scatter(triangle_points[:, 0], triangle_points[:, 1], s=5, alpha=0.5, color="green")

    for i, center in enumerate(squares):
        points_in_square = [p for p in triangle_points if point_in_square(p, center, half_size)]
        if points_in_square:
            points_arr = np.array(points_in_square)
            plt.scatter(points_arr[:, 0], points_arr[:, 1], s=8, color=colors[i], alpha=0.8)

        square_corners = np.array([
            [center[0] - half_size, center[1] - half_size],
            [center[0] + half_size, center[1] - half_size],
            [center[0] + half_size, center[1] + half_size],
            [center[0] - half_size, center[1] + half_size],
            [center[0] - half_size, center[1] - half_size]
        ])
        plt.plot(square_corners[:, 0], square_corners[:, 1], color=colors[i], linewidth=2, linestyle='--', alpha=0.7)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.28, linewidth=0.7)
    plt.title(f"Случайные точки внутри треугольника (n={n_points})")
    plt.tight_layout()
    plt.show()


def plot_circle_points(normal, radius, center, n_points):
    circle_points = np.array([
        generate_random_circle_point(normal, radius, center)
        for _ in range(n_points)
    ])

    square_half_size = 0.6
    square1_center = np.array([0.0, -0.5])
    square2_center = np.array([1.2, 0.0])
    square3_center = np.array([-0.8, 0.8])

    squares = [square1_center, square2_center, square3_center]
    colors = ['red', 'orange', 'darkred']

    def square_in_circle(center, half_size, circle_radius):
        corners = [
            [center[0] - half_size, center[1] - half_size],
            [center[0] + half_size, center[1] - half_size],
            [center[0] + half_size, center[1] + half_size],
            [center[0] - half_size, center[1] + half_size]
        ]
        return all(np.linalg.norm(corner) <= circle_radius for corner in corners)

    for i, sq_center in enumerate(squares):
        if not square_in_circle(sq_center, square_half_size, radius):
            print(f"Предупреждение: квадрат {i + 1} не полностью внутри круга!")

    counts = []
    for sq_center in squares:
        count = sum(1 for p in circle_points if point_in_square(p[:2], sq_center, square_half_size))
        counts.append(count)

    mean_count = np.mean(counts)
    deviations = [abs(c - mean_count) / mean_count * 100 for c in counts]
    max_deviation = max(deviations)

    print(f"\n=== Круг ===")
    print(f"Площадь каждого квадрата: {(2 * square_half_size) ** 2:.2f}")
    for i, count in enumerate(counts, 1):
        print(f"Количество точек в квадрате {i} (центр {squares[i - 1]}): {count}")
    print(f"Среднее количество: {mean_count:.1f}")
    print(f"Максимальное отклонение от среднего: {max_deviation:.2f}%")

    if not SHOW_PLOTS:
        print("(График пропущен: SHOW_PLOTS = False)")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], s=5, alpha=0.5, color="green")

    for i, sq_center in enumerate(squares):
        points_in_square = [p for p in circle_points if point_in_square(p[:2], sq_center, square_half_size)]
        if points_in_square:
            points_arr = np.array(points_in_square)
            ax.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], s=8, color=colors[i], alpha=0.8)

        square_corners = np.array([
            [sq_center[0] - square_half_size, sq_center[1] - square_half_size, 0],
            [sq_center[0] + square_half_size, sq_center[1] - square_half_size, 0],
            [sq_center[0] + square_half_size, sq_center[1] + square_half_size, 0],
            [sq_center[0] - square_half_size, sq_center[1] + square_half_size, 0],
            [sq_center[0] - square_half_size, sq_center[1] - square_half_size, 0]
        ])
        ax.plot(square_corners[:, 0], square_corners[:, 1], square_corners[:, 2], color=colors[i], linewidth=2,
                linestyle='--', alpha=0.7)

    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    r = np.linalg.norm(radius)

    t = np.linspace(0.0, 2.0 * np.pi, 200)
    contour = center + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], color="black", linewidth=1.5)

    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.set_title(f"Случайные точки внутри круга (n={n_points})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def plot_sphere_points(radius, center, n_points):
    sphere_points = np.array([
        generate_random_sphere_point(radius, center)
        for _ in range(n_points)
    ])

    cos_theta_max_fixed = 0.9

    axis_z = np.array([0.0, 0.0, 1.0])
    axis_x = np.array([1.0, 0.0, 0.0])
    axis_45 = np.array([1.0, 0.0, 1.0])
    axis_45 = axis_45 / np.linalg.norm(axis_45)

    circles = [
        {"center_dir": axis_z, "cos_theta_max": cos_theta_max_fixed, "name": "Полюс (0°)"},
        {"center_dir": axis_x, "cos_theta_max": cos_theta_max_fixed, "name": "Экватор (90°)"},
        {"center_dir": axis_45, "cos_theta_max": cos_theta_max_fixed, "name": "45° широта"}
    ]
    colors = ['red', 'orange', 'darkred']

    def in_spherical_circle(point, center_dir, cos_max):
        vec = point - center
        r = np.linalg.norm(vec)
        if abs(r - radius) > 0.01 * radius:
            return False
        direction = vec / r
        cos_theta = np.dot(direction, center_dir)
        return cos_theta >= cos_max

    counts = []
    for circle in circles:
        count = sum(1 for p in sphere_points
                    if in_spherical_circle(p, circle["center_dir"], circle["cos_theta_max"]))
        counts.append(count)

    mean_count = np.mean(counts)
    deviations = [abs(c - mean_count) / mean_count * 100 for c in counts]
    max_deviation = max(deviations)

    print(f"\n=== Сфера (равномерное распределение) ===")
    print(f"Площадь каждого сферического круга: {2 * np.pi * radius ** 2 * (1 - cos_theta_max_fixed):.2f}")
    for i, (circle, count) in enumerate(zip(circles, counts), 1):
        print(f"Круг {i} ({circle['name']}): {count} точек")
    print(f"Среднее количество: {mean_count:.1f}")
    print(f"Максимальное отклонение от среднего: {max_deviation:.2f}%")

    if not SHOW_PLOTS:
        print("(График пропущен: SHOW_PLOTS = False)")
        return

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, color="lightgray", alpha=0.5, linewidth=0)

    all_colored = set()
    for i, circle in enumerate(circles):
        pts = [p for p in sphere_points
               if in_spherical_circle(p, circle["center_dir"], circle["cos_theta_max"])]
        if pts:
            pts_arr = np.array(pts)
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], s=8, color=colors[i], alpha=0.8)
            for p in pts:
                all_colored.add(tuple(p))

    other = [p for p in sphere_points if tuple(p) not in all_colored]
    if other:
        other_arr = np.array(other)
        ax.scatter(other_arr[:, 0], other_arr[:, 1], other_arr[:, 2], s=3, alpha=0.3, color="green")

    ax.grid(True, alpha=0.28)
    ax.set_title(f"Случайные точки на сфере (n={n_points})\nКруги равной площади")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def plot_cosine_directions(normal, center, n_points):
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(center, dtype=float)

    directions = np.array([
        generate_random_cosine_direction(normal)
        for _ in range(n_points)
    ])
    endpoints = center + directions

    cos_theta_max_fixed = 0.9

    n = normal
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u_vec = np.cross(a, n)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(n, u_vec)

    dir_normal = n
    dir_perp = u_vec
    dir_45 = (n + u_vec) / np.linalg.norm(n + u_vec)

    circles = [
        {"center_dir": dir_normal, "cos_theta_max": cos_theta_max_fixed, "name": "Вдоль нормали (0°)"},
        {"center_dir": dir_perp, "cos_theta_max": cos_theta_max_fixed, "name": "Перпендикулярно (90°)"},
        {"center_dir": dir_45, "cos_theta_max": cos_theta_max_fixed, "name": "45° от нормали"}
    ]
    colors = ['red', 'orange', 'darkred']

    def in_spherical_circle(point, center_dir, cos_max):
        vec = point - center
        r = np.linalg.norm(vec)
        if abs(r - 1.0) > 0.01:
            return False
        direction = vec / r
        cos_theta = np.dot(direction, center_dir)
        return cos_theta >= cos_max

    counts = []
    for circle in circles:
        count = sum(1 for p in endpoints
                    if in_spherical_circle(p, circle["center_dir"], circle["cos_theta_max"]))
        counts.append(count)

    # Теоретические ожидания (нормализованные)
    cos_angles = [1.0, 0.0, np.cos(np.pi / 4)]  # cos(0°), cos(90°), cos(45°)
    expected = np.array(cos_angles) / np.sum(cos_angles) * np.sum(counts)

    # Отклонения от теоретического ожидания
    deviations = [abs(counts[i] - expected[i]) / expected[i] * 100 if expected[i] > 0 else 0
                  for i in range(len(counts))]
    max_deviation = max(deviations)

    print(f"\n=== Косинусное распределение ===")
    print("Ожидается убывание точек при удалении от нормали")
    for i, (circle, count) in enumerate(zip(circles, counts), 1):
        exp_val = expected[i - 1]
        print(f"Область {i} ({circle['name']}): {count} точек (теоретически: {exp_val:.0f})")
    print(f"Максимальное отклонение от теоретического: {max_deviation:.2f}%")

    if counts[0] > counts[1] and counts[0] > counts[2]:
        print("✓ Распределение корректное: больше точек у нормали")
    else:
        print("✗ Возможна проблема: распределение не убывает монотонно")

    if not SHOW_PLOTS:
        print("(График пропущен: SHOW_PLOTS = False)")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    radius = 1
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightgray", alpha=0.3, linewidth=0)

    all_colored = set()
    for i, circle in enumerate(circles):
        pts = [p for p in endpoints
               if in_spherical_circle(p, circle["center_dir"], circle["cos_theta_max"])]
        if pts:
            pts_arr = np.array(pts)
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], s=8, color=colors[i], alpha=0.8)
            for p in pts:
                all_colored.add(tuple(p))

    other = [p for p in endpoints if tuple(p) not in all_colored]
    if other:
        other_arr = np.array(other)
        ax.scatter(other_arr[:, 0], other_arr[:, 1], other_arr[:, 2], s=3, alpha=0.3, color="green")

    ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2],
              length=1.2, color='black', linewidth=2, arrow_length_ratio=0.2)

    lim = 1.25
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(center[2] - lim, center[2] + lim)
    ax.set_title(f"Косинусное распределение\nКруги равной площади")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n = 100000
    if SHOW_PLOTS:
        n = 5000

    triangle_vertices = np.array([
        [0.0, 1.5],
        [5.0, 0.0],
        [3.0, 3.0],
    ])

    circle_normal = np.array([0.0, 0.0, 1.0])
    circle_radius = 2.0
    circle_center = np.array([0.0, 0.0, 0.0])

    sphere_radius = 2.0
    sphere_center = np.array([0.0, 0.0, 0.0])

    cosine_normal = np.array([0.0, 0.0, 1.0])
    cosine_center = np.array([0.0, 0.0, 0.0])

    plot_triangle_points(triangle_vertices, n)
    plot_circle_points(circle_normal, circle_radius, circle_center, n)
    plot_sphere_points(sphere_radius, sphere_center, n)
    plot_cosine_directions(cosine_normal, cosine_center, n)