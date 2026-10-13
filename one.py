import math
import sympy as sp

eps = 1e-6
A = 0.0
B = 1.0

X = sp.Symbol("x")
FX = X ** 4 + sp.exp(-X)
DFX = sp.diff(FX, X)
DDFX = sp.diff(DFX, X)

print("f'(x)  =", DFX)
print("f''(x) =", DFX)
print()

f = sp.lambdify(X, FX, "math")
df = sp.lambdify(X, DFX, "math")
d2f = sp.lambdify(X, DDFX, "math")


def proverka():
    print("Проверка функции:")
    print("f'(0) =", round(df(A), 4), "должна быть < 0")
    print("f'(1) =", round(df(B), 4), "должна быть > 0")

    ok = True
    n = 1000
    for i in range(n + 1):
        x = A + (B - A) * i / n
        if d2f(x) <= 0:
            ok = False
    if ok:
        print("f''(x) > 0 на всем отрезке, значит функция выпуклая,")
        print("минимум один (функция унимодальная)")
    else:
        print("f''(x) не везде > 0, методы могут не подойти")
    print()


def pop():
    a = A
    b = B
    d = eps / 10
    n = 0
    while b - a > eps:
        x1 = (a + b - d) / 2
        x2 = (a + b + d) / 2
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        n = n + 1
    x = (a + b) / 2
    return x, f(x), n


def zolotoe():
    a = A
    b = B
    phi = (1 + math.sqrt(5)) / 2    #1.618
    t = 1 / (phi ** 2)  
    x1 = a + t * (b - a)
    x2 = b - t * (b - a)
        
    f1 = f(x1)
    f2 = f(x2)
        
    n = 2 
    while (b - a) > eps:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + t * (b - a)  #новая левая точка
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - t * (b - a)  #новая правая точка
            f2 = f(x2)
        n += 1

        if n > 1000:
            print("Превышено максимальное число итераций!")
            break
    x = (a + b) / 2
    return x, f(x), n


def lom():
    L = 0
    for i in range(1001):
        x = A + (B - A) * i / 1000
        if abs(df(x)) > L:
            L = abs(df(x))
    xs = [A, B]
    ys = [f(A), f(B)]
    n = 0
    while True:
        m = 0
        rmin = 1000000
        for i in range(len(xs) - 1):
            r = (ys[i] + ys[i + 1]) / 2 - L * (xs[i + 1] - xs[i]) / 2
            if r < rmin:
                rmin = r
                m = i
        best = min(ys)
        if best - rmin < eps:
            break
        xn = (xs[m] + xs[m + 1]) / 2 + (ys[m] - ys[m + 1]) / (2 * L)
        xs.insert(m + 1, xn)
        ys.insert(m + 1, f(xn))
        n = n + 1
    i = ys.index(min(ys))
    return xs[i], ys[i], n


def kas():
    a = A
    b = B
    n = 0
    while b - a > eps:
        xm = (f(b) - f(a) + df(a) * a - df(b) * b) / (df(a) - df(b))
        g = df(xm)
        if abs(g) < eps:
            a = xm
            b = xm
        elif g > 0:
            b = xm
        else:
            a = xm
        n = n + 1
    x = (a + b) / 2
    return x, f(x), n


def nyuton():
    x = B
    n = 0
    while True:
        xk = x - df(x) / d2f(x)
        n = n + 1
        if abs(xk - x) < eps:
            x = xk
            break
        x = xk
    return x, f(x), n


proverka()

print("Результаты (eps = 0.000001):")
print()

x, fx, n = pop()
print("Деление пополам:   x = %.6f  f = %.6f  итераций = %d" % (x, fx, n))

x, fx, n = zolotoe()
print("Золотое сечение:   x = %.6f  f = %.6f  итераций = %d" % (x, fx, n))

x, fx, n = lom()
print("Метод ломаных:     x = %.6f  f = %.6f  итераций = %d" % (x, fx, n))

x, fx, n = kas()
print("Метод касательных: x = %.6f  f = %.6f  итераций = %d" % (x, fx, n))

x, fx, n = nyuton()
print("Метод Ньютона:     x = %.6f  f = %.6f  итераций = %d" % (x, fx, n))