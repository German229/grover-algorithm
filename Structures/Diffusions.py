from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray

from Structures.Gate import GenericGate


class Diffusion(ABC):
    """
    Абстрактный базовый класс для диффузионных операторов,
    используемых в алгоритме Гровера.

    Требует реализации методов для получения матрицы оператора
    и преобразования её в объект GenericGate.
    """

    @abstractmethod
    def get_matrix(self) -> ndarray:
        """
        Возвращает матрицу диффузионного оператора в виде массива NumPy.

        Метод должен быть реализован в наследуемом классе.

        Returns:
            ndarray: унитарная матрица комплексных чисел размером 2^n на 2^n.
        """
        pass

    @abstractmethod
    def to_gate(self) -> GenericGate:
        """
        Преобразует матрицу диффузионного оператора в объект класса GenericGate.

        Метод должен быть реализован в наследуемом классе.

        Returns:
            GenericGate: объект, содержащий матрицу оператора.
        """
        pass


class GenericDiffusion(Diffusion):
    """
    Базовая реализация диффузионного оператора для n кубитов.

    Реализует метод построения матрицы отражения относительно вектора psi:
        D = 2 * psi * psi^dagger - I

    Абстрактный метод get_psi() должен возвращать вектор состояния psi
    для конкретной реализации отражения.
    """

    def __init__(self, num_qubits: int):
        """
        Инициализирует объект для построения диффузионного оператора.

        Parameters:
            num_qubits (int): число кубитов в регистре.
        """
        self.n = num_qubits

    @abstractmethod
    def get_psi(self) -> ndarray:
        """
        Возвращает вектор psi (состояние), относительно которого происходит отражение.

        Метод должен быть реализован в конкретной реализации диффузии.

        Returns:
            ndarray: нормированный вектор размером 2^n на 1.
        """
        pass

    def get_matrix(self) -> ndarray:
        """
        Строит матрицу диффузионного оператора:
            D = 2 * psi * psi^dagger - I

        Returns:
            ndarray: квадратная комплексная матрица 2^n на 2^n.
        """
        size = 2 ** self.n
        psi = self.get_psi()
        projector = 2 * (psi @ psi.T.conj())
        identity = np.eye(size, dtype=complex)
        return projector - identity

    def to_gate(self) -> GenericGate:
        """
        Преобразует построенную матрицу диффузии в объект GenericGate.

        Returns:
            GenericGate: гейт с матрицей оператора отражения.
        """
        return GenericGate(self.get_matrix())


class StandardDiffusion(GenericDiffusion):
    """
    Стандартная реализация диффузионного оператора для алгоритма Гровера.

    Вектор psi - равномерная суперпозиция всех базисных состояний:
        psi = (1 / sqrt(2^n)) * [1, 1, ..., 1]^T
    """

    def __init__(self, num_qubits: int):
        """
        Инициализирует стандартный диффузионный оператор.

        Parameters:
            num_qubits (int): число кубитов в регистре.
        """
        super().__init__(num_qubits)

    def get_psi(self) -> ndarray:
        """
        Возвращает нормированный вектор psi - равномерную суперпозицию всех базисов.

        Returns:
            ndarray: вектор размера 2^n на 1 с элементами (1 / sqrt(2^n)).
        """
        size = 2 ** self.n
        return np.ones((size, 1), dtype=complex) / np.sqrt(size)
