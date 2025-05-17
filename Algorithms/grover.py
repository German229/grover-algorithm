import numpy as np

from Structures.Registers import QuantumRegister
from Structures.Gate import Gate_H
from Structures.Diffusions import StandardDiffusion
from Structures.Oracle import Oracle

def run_grover(oracle : Oracle) -> int:
    """
    Выполняет алгоритм Гровера для поиска решения задачи вида f(x) = 1,
    используя только линейную алгебру.

    Алгоритм выполняет следующие шаги:
    1. Инициализирует квантовый регистр в состоянии |00...0>.
    2. Применяет оператор Адамара к каждому кубиту (формирует суперпозицию).
    3. Применяет оракул, отражающий амплитуду целевых состояний.
    4. Применяет диффузионный оператор (отражение относительно средней амплитуды).
    5. Измеряет состояние регистра и возвращает результат в виде числа x.

    Параметры:
        oracle (Oracle): объект оракула, реализующий метод to_gate(),
                         возвращающий матрицу унитарного преобразования.

    Возвращает:
        int: число x от 0 до 2**n - 1, где f(x) = 1 с высокой вероятностью.
    """
    oracle_gate = oracle.to_gate()
    n = int(np.log2(oracle_gate.gate_matrix.shape[0]))
    reg = QuantumRegister(n)

    h = Gate_H()
    h_total = h
    for _ in range(n - 1):
        h_total = h_total.tensor(h)
    reg.apply_gate(h_total.gate_matrix)

    reg.apply_gate(oracle_gate.gate_matrix)

    diffusion = StandardDiffusion(n)
    reg.apply_gate(diffusion.to_gate().gate_matrix)


    return reg.measure()

