import numpy as np
import matplotlib.pyplot as plt
from Structures.Gate import Gate_H
from Structures.Oracle import OracleAND
from Structures.Diffusions import StandardDiffusion


def analyze_probability(max_iterations=5, verbose=True):
    """
    Выводит, как изменяется вероятность получения состояния |11⟩
    на каждом шаге алгоритма Гровера.
    """
    h_gate = Gate_H()
    oracle = OracleAND()
    diffuser = StandardDiffusion(num_qubits=2)

    state = np.array([[1], [0], [0], [0]], dtype=complex)

    # Применяем H ⊗ H
    H2 = h_gate.tensor(h_gate)
    state = H2.gate_matrix @ state

    results = []
    for i in range(max_iterations + 1):
        prob = np.abs(state[3, 0]) ** 2  # вероятность |11⟩
        results.append(prob)
        if verbose:
            print(f"Итерация {i}: вероятность |11⟩ = {prob:.4f}")

        # Один шаг Гровера: оракул + диффузия
        state = oracle.get_matrix() @ state
        state = diffuser.get_matrix() @ state

    return results


def plot_probability(max_iterations=5):
    results = analyze_probability(max_iterations=max_iterations, verbose=False)
    plt.plot(range(len(results)), results, marker='o')
    plt.title("Рост вероятности состояния |11⟩ в алгоритме Гровера")
    plt.xlabel("Номер итерации")
    plt.ylabel("Вероятность состояния |11⟩")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
