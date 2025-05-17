from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Register(ABC):
    """
    Абстрактный базовый класс для квантового регистра.

    Определяет интерфейс для получения состояния, применения гейта
    и измерения состояния квантовой системы из нескольких кубитов.
    """

    def __init__(self, num_qubits: int):
        """
        Инициализирует квантовый регистр на заданном количестве кубитов.

        Parameters:
            num_qubits (int): количество кубитов в регистре.
        """
        self.num_qubits = num_qubits

    @abstractmethod
    def get_state(self) -> ndarray:
        """
        Возвращает текущее состояние квантового регистра.

        Returns:
            ndarray: вектор-столбец комплексных амплитуд длины 2^n.
        """
        pass

    @abstractmethod
    def apply_gate(self, gate_matrix: ndarray) -> None:
        """
        Применяет унитарный оператор (гейт) к текущему состоянию регистра.

        Parameters:
            gate_matrix (ndarray): квадратная унитарная матрица размера 2^n на 2^n.
        """
        pass

    @abstractmethod
    def measure(self) -> int:
        """
        Выполняет измерение регистра в стандартном вычислительном базисе.

        Returns:
            int: индекс состояния, выбранного случайно с вероятностью,
                 пропорциональной квадрату модуля амплитуды.
        """
        pass


class GenericRegister(Register):
    """
    Базовая реализация квантового регистра.

    Хранит состояние регистра в виде нормированного комплексного вектора
    и реализует методы для применения гейтов и измерения.
    """

    def __init__(self, num_qubits: int):
        """
        Инициализирует регистр на n кубитах и устанавливает начальное состояние |00...0>.

        Parameters:
            num_qubits (int): количество кубитов.
        """
        super().__init__(num_qubits)
        self.state = np.zeros((2 ** num_qubits, 1), dtype=complex)
        self.state[0][0] = 1  # начальное состояние — только |00...0> с амплитудой 1

    def get_state(self) -> ndarray:
        """
        Возвращает текущее состояние регистра.

        Returns:
            ndarray: вектор амплитуд (размерность 2^n на 1).
        """
        return self.state

    def apply_gate(self, gate_matrix: ndarray) -> None:
        """
        Применяет гейт к текущему состоянию регистра.

        Parameters:
            gate_matrix (ndarray): матрица квантового гейта размера 2^n на 2^n.
        """
        self.state = gate_matrix @ self.state

    def measure(self) -> int:
        """
        Выполняет вероятностное измерение регистра.

        Returns:
            int: индекс одного из базисных состояний, выбранный согласно
                 распределению вероятностей по квадратам модулей амплитуд.
        """
        probabilities = np.abs(self.state.flatten()) ** 2
        return int(np.random.choice(len(probabilities), p=probabilities))


class QuantumRegister(GenericRegister):
    """
    Специализированная реализация квантового регистра.

    Наследует поведение GenericRegister и используется как конкретный
    экземпляр регистра в алгоритмах (например, Гровера).
    """

    def __init__(self, num_qubits: int):
        """
        Инициализирует квантовый регистр с заданным числом кубитов.

        Parameters:
            num_qubits (int): количество кубитов.
        """
        super().__init__(num_qubits)
