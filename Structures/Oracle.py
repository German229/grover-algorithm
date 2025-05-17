from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from Structures.Gate import GenericGate


class Oracle(ABC):
    """
    Абстрактный базовый класс для оракулов, используемых в алгоритме Гровера.

    Оракул реализует булеву функцию f(x) и возвращает унитарную матрицу,
    которая инвертирует амплитуду тех состояний, для которых f(x) = 1.
    """

    @abstractmethod
    def get_matrix(self) -> ndarray:
        """
        Возвращает унитарную матрицу оракула U_f, которая отражает амплитуды
        базисных состояний, соответствующих f(x) = 1.

        Returns:
            ndarray: квадратная комплексная матрица размера 2^n на 2^n.
        """
        pass

    @abstractmethod
    def get_function(self):
        """
        Возвращает классическую булеву функцию f(x), реализуемую данным оракулом.

        Returns:
            Callable[[int], int]: функция f: {0, ..., 2^n - 1} → {0, 1}.
        """
        pass

    @abstractmethod
    def to_gate(self) -> GenericGate:
        """
        Оборачивает матрицу оракула в объект GenericGate.

        Returns:
            GenericGate: объект, содержащий матрицу оракула.
        """
        pass


class GenericOracle(Oracle):
    """
    Обобщённая реализация оракула, основанная на заданной функции f(x)
    и числе кубитов n.

    Использует диагональную матрицу, где элемент с индексом i равен (-1)^f(i).
    """

    def __init__(self, n_qubits: int):
        """
        Инициализирует оракул на n кубитах.

        Parameters:
            n_qubits (int): количество кубитов.
        """
        self.n = n_qubits

    @abstractmethod
    def get_function(self):
        """
        Возвращает классическую булеву функцию f(x), реализуемую данным оракулом.

        Returns:
            Callable[[int], int]: функция f(x), определяющая, какие состояния инвертировать.
        """
        pass

    def get_matrix(self) -> ndarray:
        """
        Строит диагональную матрицу оракула:
            U_f = diag((-1)^f(0), (-1)^f(1), ..., (-1)^f(2^n - 1))

        Returns:
            ndarray: комплексная диагональная матрица размера 2^n на 2^n.
        """
        size = 2 ** self.n
        f = self.get_function()
        diag = [(-1) ** f(i) for i in range(size)]
        return np.diag(diag).astype(complex)

    def to_gate(self) -> GenericGate:
        """
        Возвращает объект GenericGate, содержащий матрицу оракула.

        Returns:
            GenericGate: гейт с матрицей U_f.
        """
        return GenericGate(self.get_matrix())


class OracleAND(GenericOracle):
    """
    Конкретная реализация оракула для функции f(x) = x_0 AND x_1,
    где x — двухбитное значение, представляемое числом от 0 до 3 (n = 2).
    """

    def __init__(self):
        """
        Инициализирует оракул на 2 кубитах для функции f(x) = (x >> 1) & 1 AND x & 1.
        """
        super().__init__(2)

    def get_function(self):
        """
        Возвращает булеву функцию f(x) = (x_0 AND x_1), реализуемую как:
            f(x) = ((x >> 1) & 1) AND (x & 1)

        Returns:
            Callable[[int], int]: функция f(x), равная 1 только при x = 3.
        """
        return lambda x: ((x >> 1) & 1) & (x & 1)
