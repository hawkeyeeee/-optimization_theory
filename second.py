import math

EPS = 1e-6
x0 = (1.0, 1.0)

def f(x1, x2):
    root_part = math.sqrt(1 + 2 * x1 ** 2 + x2 ** 2)
    exp_part = math.exp(x1 ** 2 + 2 * x2 ** 2)
    return root_part + exp_part - x1 - x2

def safe_f(x1, x2):
    try:
        return f(x1, x2)
    except OverflowError:
        return float("inf")

def gradient(x1, x2):
    s = math.sqrt(1 + 2 * x1 ** 2 + x2 ** 2)
    e = math.exp(x1 ** 2 + 2 * x2 ** 2)
    df_dx1 = 2 * x1 / s + 2 * x1 * e - 1
    df_dx2 = x2 / s + 4 * x2 * e - 1
    return df_dx1, df_dx2

def hessian(x1, x2):
    s = math.sqrt(1 + 2 * x1 ** 2 + x2 ** 2)
    e = math.exp(x1 ** 2 + 2 * x2 ** 2)
    h11 = 2 / s - 4 * x1 ** 2 / s ** 3 + (2 + 4 * x1 ** 2) * e
    h12 = -2 * x1 * x2 / s ** 3 + 8 * x1 * x2 * e
    h22 = 1 / s - x2 ** 2 / s ** 3 + (4 + 16 * x2 ** 2) * e
    return h11, h12, h22

def check_convexity():
    print("Проверка применимости методов (матрица Гессе)")
    print("Матрица Гессе положительно определена, если h11 > 0 и h11*h22 - h12^2 > 0")
    test_points = [(1, 1), (0, 0), (-1, 1), (2, -2), (-3, 0.5), (0.3, 0.2), (0, -1)]
    all_ok = True
    for x1, x2 in test_points:
        h11, h12, h22 = hessian(x1, x2)
        det = h11 * h22 - h12 ** 2
        ok = h11 > 0 and det > 0
        if not ok:
            all_ok = False
        print("точка", (x1, x2), "h11 =", round(h11, 4), "det =", round(det, 4), "ок" if ok else "НЕ ок")
    if all_ok:
        print("Во всех проверенных точках Гессиан положительно определен,")
        print("функция выпуклая, минимум единственный, методы применимы")
    print()

def nelder_mead(start, eps):
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    delta = 0.5
    step = 1.0

    p1 = [start[0], start[1]]
    p2 = [start[0] + step, start[1]]
    p3 = [start[0], start[1] + step]
    points = [p1, p2, p3]

    iteration = 0
    while True:
        values = [f(p[0], p[1]) for p in points]

        order = sorted(range(3), key=lambda i: values[i])
        points = [points[i] for i in order]
        values = [values[i] for i in order]

        best = points[0]
        good = points[1]
        worst = points[2]
        f_best = values[0]
        f_good = values[1]
        f_worst = values[2]

        mean_value = (f_best + f_good + f_worst) / 3
        sum_sq = 0.0
        for v in values:
            sum_sq += (v - mean_value) ** 2
        sigma = math.sqrt(sum_sq / 3)
        if sigma < eps:
            break

        iteration += 1

        center_x1 = (best[0] + good[0]) / 2
        center_x2 = (best[1] + good[1]) / 2

        reflect_x1 = center_x1 + alpha * (center_x1 - worst[0])
        reflect_x2 = center_x2 + alpha * (center_x2 - worst[1])
        f_reflect = f(reflect_x1, reflect_x2)

        if f_reflect < f_best:
            expand_x1 = center_x1 + gamma * (reflect_x1 - center_x1)
            expand_x2 = center_x2 + gamma * (reflect_x2 - center_x2)
            f_expand = f(expand_x1, expand_x2)
            if f_expand < f_reflect:
                points[2] = [expand_x1, expand_x2]
            else:
                points[2] = [reflect_x1, reflect_x2]

        elif f_reflect < f_good:
            points[2] = [reflect_x1, reflect_x2]

        else:
            if f_reflect < f_worst:
                base_x1 = reflect_x1
                base_x2 = reflect_x2
                f_base = f_reflect
            else:
                base_x1 = worst[0]
                base_x2 = worst[1]
                f_base = f_worst

            contract_x1 = center_x1 + beta * (base_x1 - center_x1)
            contract_x2 = center_x2 + beta * (base_x2 - center_x2)
            f_contract = f(contract_x1, contract_x2)

            if f_contract < f_base:
                points[2] = [contract_x1, contract_x2]
            else:
                for i in (1, 2):
                    points[i][0] = best[0] + delta * (points[i][0] - best[0])
                    points[i][1] = best[1] + delta * (points[i][1] - best[1])

    return points[0][0], points[0][1], iteration

def gradient_descent(start, eps):
    x1 = start[0]
    x2 = start[1]
    iteration = 0

    while True:
        g1, g2 = gradient(x1, x2)
        grad_norm = math.sqrt(g1 ** 2 + g2 ** 2)
        if grad_norm < eps:
            break

        iteration += 1

        step = 1.0
        current_value = f(x1, x2)
        while True:
            new_x1 = x1 - step * g1
            new_x2 = x2 - step * g2
            new_value = safe_f(new_x1, new_x2)
            if new_value <= current_value - 0.5 * step * grad_norm ** 2:
                break
            step = step / 2

        x1 = new_x1
        x2 = new_x2

    return x1, x2, iteration

check_convexity()

x1, x2, n = nelder_mead(x0, EPS)
print("Метод Нелдера-Мида (нулевой порядок)")
print("точка минимума: x1 =", round(x1, 6), " x2 =", round(x2, 6))
print("минимальное значение функции:", round(f(x1, x2), 8))
print("количество итераций:", n)
print()

x1, x2, n = gradient_descent(x0, EPS)
print("Метод градиентного спуска (первый порядок)")
print("точка минимума: x1 =", round(x1, 6), " x2 =", round(x2, 6))
print("минимальное значение функции:", round(f(x1, x2), 8))
print("количество итераций:", n)
print("норма градиента в найденной точке:", math.sqrt(sum(g ** 2 for g in gradient(x1, x2))))
        