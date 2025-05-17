import numpy as np
from numpy import ndarray


class Qubit:
    """
    Класс Qubit представляет собой один кубит — нормированный вектор
    размерности 2, описывающий квантовое состояние.

    Состояние хранится в виде вектора-столбца:
        [alpha, beta]^T,
    где alpha и beta — комплексные амплитуды.

    При инициализации автоматически выполняется нормализация вектора.
    """

    def __init__(self, alfa_beta: ndarray) -> None:
        """
        Инициализирует кубит по переданному вектору амплитуд.

        Parameters:
            alfa_beta (ndarray): массив из двух чисел (возможно вещественных),
                                 задающих амплитуды состояний |0> и |1>.
        """
        self.alfa_beta = alfa_beta.astype(complex)
        self._normalize()

    def _normalize(self) -> None:
        """
        Нормализует вектор состояния так, чтобы сумма квадратов модулей амплитуд
        равнялась 1. Нормализация обязательна для физически корректного кубита.
        """
        norm = np.linalg.norm(self.alfa_beta)
        if norm != 0:
            self.alfa_beta /= norm

    def apply_gate(self, gate_matrix: ndarray) -> "Qubit":
        """
        Применяет квантовый гейт (матрицу 2 на 2) к текущему состоянию кубита.

        Parameters:
            gate_matrix (ndarray): унитарная матрица размера 2 на 2.

        Returns:
            Qubit: новый кубит с обновлённым состоянием.
        """
        return Qubit(gate_matrix @ self.alfa_beta)

    def get_state(self) -> ndarray:
        """
        Возвращает текущее состояние кубита в виде нормированного вектора.

        Returns:
            ndarray: вектор-столбец [alpha, beta]^T.
        """
        return self.alfa_beta

    def __repr__(self) -> str:
        """
        Возвращает строковое представление кубита в виде линейной комбинации |0> и |1>.

        Returns:
            str: строка вида "|Psi> = a|0> + b|1>", где a и b — амплитуды.
        """
        a, b = self.alfa_beta.flatten()
        return f"|Psi> = {a:.2f}|0> + {b:.2f}|1>"
