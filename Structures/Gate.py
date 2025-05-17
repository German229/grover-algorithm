from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray

from LinAl_Sem_2_Lab_3.Structures.Qubit import Qubit


class Gate(ABC):
    """
    Абстрактный базовый класс для квантовых логических гейтов.

    Описывает интерфейс для операций над кубитами, проверку унитарности
    и тензорное произведение гейтов.
    """

    def __init__(self, gate_matrix: ndarray) -> None:
        """
        Инициализирует гейт с заданной унитарной матрицей.

        Parameters:
            gate_matrix (ndarray): квадратная комплексная матрица гейта.
        """
        self.gate_matrix = gate_matrix

    @abstractmethod
    def apply_to(self, qubit: Qubit) -> Qubit:
        """
        Применяет гейт к заданному кубиту и возвращает новый кубит.

        Parameters:
            qubit (Qubit): исходный кубит.

        Returns:
            Qubit: новый кубит после применения гейта.
        """
        pass

    @abstractmethod
    def tensor(self, other_gate: "Gate") -> "Gate":
        """
        Возвращает новый гейт, являющийся тензорным произведением
        текущего гейта и другого гейта.

        Parameters:
            other_gate (Gate): другой гейт.

        Returns:
            Gate: новый гейт, полученный тензорным произведением.
        """
        pass

    @abstractmethod
    def is_unitary(self) -> bool:
        """
        Проверяет, является ли матрица гейта унитарной.

        Returns:
            bool: True, если матрица унитарна, иначе False.
        """
        pass

    @abstractmethod
    def __matmul__(self, other) -> "Gate":
        """
        Возвращает новый гейт, являющийся произведением текущего гейта и другого.

        Parameters:
            other (Gate): другой гейт.

        Returns:
            Gate: результат матричного умножения.
        """
        pass


class GenericGate(Gate):
    """
    Общая реализация квантового гейта. Поддерживает применение,
    тензорное произведение, проверку унитарности и композицию.
    """

    def __init__(self, gate_matrix: ndarray):
        """
        Инициализирует гейт с заданной матрицей.

        Parameters:
            gate_matrix (ndarray): унитарная матрица гейта.
        """
        super().__init__(gate_matrix)

    def apply_to(self, qubit: Qubit) -> Qubit:
        """
        Применяет гейт к кубиту.

        Returns:
            Qubit: результат действия гейта.
        """
        return Qubit(self.gate_matrix @ qubit.get_state())

    def tensor(self, other_gate: "Gate") -> "Gate":
        """
        Выполняет тензорное произведение двух гейтов.

        Returns:
            GenericGate: новый гейт.
        """
        return GenericGate(np.kron(self.gate_matrix, other_gate.gate_matrix))

    def is_unitary(self) -> bool:
        """
        Проверяет унитарность матрицы гейта.

        Returns:
            bool: True, если унитарна.
        """
        identity = np.eye(self.gate_matrix.shape[0], dtype=complex)
        product = self.gate_matrix.conj().T @ self.gate_matrix
        return np.allclose(product, identity)

    def __matmul__(self, other: "Gate") -> "Gate":
        """
        Выполняет последовательное применение двух гейтов (матричное умножение).

        Returns:
            GenericGate: новый гейт.
        """
        return GenericGate(self.gate_matrix @ other.gate_matrix)


class Gate_X(GenericGate):
    """
    Гейт Паули-X. Переводит |0> в |1> и |1> в |0> (аналог классического NOT).
    """

    def __init__(self):
        x_matrix = np.array([[0, 1],
                             [1, 0]], dtype=complex)
        super().__init__(x_matrix)


class Gate_H(GenericGate):
    """
    Гейт Адамара. Переводит базисные состояния в суперпозиции.
    """

    def __init__(self):
        h_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        super().__init__(h_matrix)

    def superposition(self) -> Qubit:
        """
        Применяет гейт Адамара к состоянию |0> и возвращает суперпозицию.

        Returns:
            Qubit: результат применения H к |0>.
        """
        base = Qubit(np.array([1, 0]))
        return self.apply_to(base)


class Gate_Z(GenericGate):
    """
    Гейт Паули-Z. Меняет знак у амплитуды состояния |1>, оставляя |0> без изменений.
    """

    def __init__(self):
        z_matrix = np.array([[1, 0],
                             [0, -1]], dtype=complex)
        super().__init__(z_matrix)
