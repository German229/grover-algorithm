from Structures.Gate import Gate_H
import numpy as np
from Structures.Oracle import OracleAND
from Structures.Registers import QuantumRegister
from Structures.Diffusions import StandardDiffusion
from Algorithms.grover import run_grover


def test_superposition(verbose=True):
    """
    Проверяет, что применение гейта Адамара к каждому из двух кубитов (H ⊗ H)
    к начальному состоянию |00> приводит к равномерной суперпозиции.

    Ожидается вектор состояния:
        (1/2) * [1, 1, 1, 1]^T

    Parameters:
        verbose (bool): если True - печатает амплитуды каждого базисного состояния.
    """
    h = Gate_H()
    h2 = h.tensor(h)
    reg = QuantumRegister(2)
    reg.apply_gate(h2.gate_matrix)
    result = reg.get_state()
    expected = np.ones((4, 1), dtype=complex) / 2

    if verbose:
        print("Результат применения H⊗H к |00>:")
        for i, a in enumerate(result.flatten()):
            print(f"  |{i:02b}>: амплитуда = {a.real:.2f} + {a.imag:.2f}j")

    assert np.allclose(result, expected), "Ошибка: суперпозиция H⊗H построена неверно."


def test_oracle_and(verbose=True):
    """
    Проверяет, что оракул OracleAND возвращает корректную диагональную матрицу,
    в которой только амплитуда состояния |11> (x = 3) инвертирована (равна -1).

    Parameters:
        verbose (bool): если True - выводит матрицу оракула.
    """
    oracle = OracleAND()
    matrix = oracle.get_matrix()
    expected = np.diag([1, 1, 1, -1])

    if verbose:
        print("Матрица оракула OracleAND:")
        print(matrix)

    assert np.allclose(matrix, expected), "Ошибка: OracleAND неверно инвертирует |11>."


def test_diffusion_matrix(verbose=True):
    """
    Проверяет, что диффузионный оператор D корректно отражает состояние
    равномерной суперпозиции psi относительно самого себя.

    То есть должно выполняться:
        D * psi = psi

    Parameters:
        verbose (bool): если True - выводит амплитуды состояния после отражения.
    """
    diffusion = StandardDiffusion(2)
    D = diffusion.get_matrix()
    psi = np.ones((4, 1), dtype=complex) / 2
    reflected = D @ psi

    if verbose:
        print("Результат применения D к равномерному состоянию psi:")
        for i, a in enumerate(reflected.flatten()):
            print(f"  |{i:02b}>: амплитуда = {a.real:.2f} + {a.imag:.2f}j")

    assert np.allclose(reflected, psi), "Ошибка: оператор D искажает состояние psi."


def test_run_grover(verbose=True):
    """
    Проверяет, что алгоритм Гровера с оракулом AND успешно находит состояние |11> (x = 3)
    с высокой вероятностью. Тест считается успешным, если не менее 8 из 10 измерений
    дают результат x = 3.

    Parameters:
        verbose (bool): если True - выводит каждый результат измерения.
    """
    hits = 0
    attempts = 20
    for i in range(attempts):
        result = run_grover(OracleAND())
        if verbose:
            print(f"Измерение {i + 1}: результат = {result}")
        if result == 3:
            hits += 1

    assert hits >= attempts * 0.8, f"Ошибка: недостаточно попаданий в |11>. Успехов: {hits}/attempts"
