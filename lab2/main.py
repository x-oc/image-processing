from pathlib import Path

import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

rng = np.random.default_rng(0)


def calculate_simple_monte_carlo_integral(a, b, function, n):
    samples = rng.uniform(a, b, size=n)
    y = function(samples)
    return np.mean(y) * (b - a)


def calculate_stratified_monte_carlo_integral(a, b, function, step, n):
    partition = np.arange(a, b + 1e-6, step)
    result = 0
    for i in range(len(partition) - 1):
        samples = rng.uniform(partition[i], partition[i + 1], size=n)
        result += np.mean(function(samples)) * (partition[i + 1] - partition[i])
    return result


def calculate_importance_monte_carlo_integral(function, pdf, inv_cdf, n):
    samples = inv_cdf(rng.uniform(0, 1, size=n))
    y = function(samples) / pdf(samples)
    return np.mean(y)


def calculate_multi_importance_monte_carlo_integral(function, pdfs, inv_cdfs, weight_functions, n):
    base_samples = rng.uniform(0, 1, size=n)
    result = 0
    for pdf, inv_cdf, weight_function in zip(pdfs, inv_cdfs, weight_functions):
        samples = inv_cdf(base_samples)
        result += function(samples) / pdf(samples) * weight_function(samples, pdfs[0], pdfs[1])
    return np.mean(result)


def calculate_russian_roulette_monte_carlo_integral(function, pdf, inv_cdf, r, n):
    y = 0
    for _ in range(n):
        sample = inv_cdf(rng.uniform(0, 1))
        if rng.uniform(0, 1) < r:
            y += function(sample) / (pdf(sample) * r)
    return y / n


square = lambda x: x ** 2
square_integral = lambda x: x ** 3 / 3

# функции плотности вероятности, домноженнные на нормировочный коэффициент
p1 = lambda x: x * 2 / 21
p2 = lambda x: x ** 2 / 39
p3 = lambda x: x ** 3 * 4 / 609

# обратные функции распределения для генерации случайных чисел
f1 = lambda x: (x * 21 + 4) ** (1 / 2)
f2 = lambda x: (x * 117 + 8) ** (1 / 3)
f3 = lambda x: (x * 609 + 16) ** (1 / 4)

# веса по среднему арифметическому и среднему квадрату плотностей
w1 = lambda x, pdf1, pdf2: pdf1(x) / (pdf1(x) + pdf2(x))
w2 = lambda x, pdf1, pdf2: pdf2(x) / (pdf1(x) + pdf2(x))
w3 = lambda x, pdf1, pdf2: pdf1(x) ** 2 / (pdf1(x) ** 2 + pdf2(x) ** 2)
w4 = lambda x, pdf1, pdf2: pdf2(x) ** 2 / (pdf1(x) ** 2 + pdf2(x) ** 2)

r1 = 0.5
r2 = 0.75
r3 = 0.95

n1 = 100
n2 = 1000
n3 = 10000
n4 = 100000

a, b = 2, 5

true_integral_value = square_integral(5) - square_integral(2)


def run_experiments(a, b, true_integral_value):
    results = []

    for n in [n1, n2, n3, n4]:
        estimate = calculate_simple_monte_carlo_integral(a, b, square, n)
        results.append(
            {
                "method": "simple",
                "n": n,
                "params": "-",
                "estimate": estimate,
                "abs_error": abs(estimate - true_integral_value),
            }
        )

        for step in [1, 0.5]:
            estimate = calculate_stratified_monte_carlo_integral(a, b, square, step, n)
            results.append(
                {
                    "method": "stratified",
                    "n": n,
                    "params": f"step={step}",
                    "estimate": estimate,
                    "abs_error": abs(estimate - true_integral_value),
                }
            )

        for pdf_name, pdf, inv_cdf in zip(["p1", "p2", "p3"], [p1, p2, p3], [f1, f2, f3]):
            estimate = calculate_importance_monte_carlo_integral(square, pdf, inv_cdf, n)
            results.append(
                {
                    "method": "importance",
                    "n": n,
                    "params": f"pdf={pdf_name}",
                    "estimate": estimate,
                    "abs_error": abs(estimate - true_integral_value),
                }
            )

            for r in [r1, r2, r3]:
                estimate = calculate_russian_roulette_monte_carlo_integral(square, pdf, inv_cdf, r, n)
                results.append(
                    {
                        "method": "russian_roulette",
                        "n": n,
                        "params": f"pdf={pdf_name}, r={r}",
                        "estimate": estimate,
                        "abs_error": abs(estimate - true_integral_value),
                    }
                )

        for weight_name, weight_function_pair in [("balance", [w1, w2]), ("power", [w3, w4])]:
            estimate = calculate_multi_importance_monte_carlo_integral(
                square, [p1, p3], [f1, f3], weight_function_pair, n
            )
            results.append(
                {
                    "method": "multi_importance",
                    "n": n,
                    "params": f"pdfs=p1,p3; weights={weight_name}",
                    "estimate": estimate,
                    "abs_error": abs(estimate - true_integral_value),
                }
            )

    return results


def build_table_string(headers, table_rows):
    if tabulate is not None:
        return tabulate(table_rows, headers=headers, tablefmt="fancy_grid")

    row_template = "{:<8} {:<26} {:>12} {:>12}"
    lines = [row_template.format(*headers), "-" * 75]
    for row in table_rows:
        lines.append(row_template.format(*row))
    return "\n".join(lines)


def write_results_tables_to_file(results, output_path):
    headers = ["n", "params", "estimate", "abs_error"]
    preferred_order = ["simple", "stratified", "importance", "multi_importance", "russian_roulette"]
    methods = [method for method in preferred_order if any(row["method"] == method for row in results)]
    sections = []
    summary_rows = []

    for method in methods:
        method_rows = [row for row in results if row["method"] == method]
        method_rows.sort(key=lambda row: (row["n"], row["params"]))

        table_rows = []
        for row in method_rows:
            table_rows.append(
                [
                    row["n"],
                    row["params"],
                    f"{row['estimate']:.6f}",
                    f"{row['abs_error']:.6f}",
                ]
            )

        section_title = f"Method: {method}"
        sections.append(f"{section_title}\n{build_table_string(headers, table_rows)}")

        best_row = min(method_rows, key=lambda row: row["abs_error"])
        summary_rows.append(
            [
                method,
                best_row["n"],
                best_row["params"],
                f"{best_row['estimate']:.6f}",
                f"{best_row['abs_error']:.6f}",
            ]
        )

    summary_headers = ["method", "n", "params", "estimate", "abs_error"]
    sections.append(f"Summary: best result per method\n{build_table_string(summary_headers, summary_rows)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections), encoding="utf-8")


results = run_experiments(a, b, true_integral_value)
output_file = Path(__file__).parent / "output" / "output.txt"
write_results_tables_to_file(results, output_file)
