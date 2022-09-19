"""
This project is to verify the independent approximation,
how that will influence the stationary distribution


"""
import numpy as np
import matplotlib.pyplot as plt


class SingleQueue:
    def __init__(self, max_length=100):
        self.dim = max_length
        self.pmf_list = [1] + [0 for _ in range(max_length - 1)]

    def arrival_step(self, arrival_prob=0.2):
        # todo: overall flow is not dealt well
        arrival_prob = max(min(arrival_prob, 1), 0)
        no_arrival_list = self.pmf_list + [0]
        with_arrival_list = [0] + self.pmf_list
        new_pmf_list = np.array(no_arrival_list) * (1 - arrival_prob)
        new_pmf_list += np.array(with_arrival_list) * arrival_prob
        new_pmf_list = new_pmf_list.tolist()
        self.pmf_list = new_pmf_list[: self.dim]
        return self.pmf_list

    def departure_step(self, departure_prob=0.0):
        departure_prob = max(min(departure_prob, 1), 0)
        no_residual_prob = self.pmf_list[0]
        with_departure_list = self.pmf_list[1:] + [0]
        with_departure_list[0] += no_residual_prob
        no_departure_list = self.pmf_list
        new_pmf_list = np.array(with_departure_list) * departure_prob
        new_pmf_list += np.array(no_departure_list) * (1 - departure_prob)
        new_pmf_list = new_pmf_list.tolist()
        self.pmf_list = new_pmf_list[: self.dim]
        return (1 - no_residual_prob) * departure_prob


class JointQueue:
    def __init__(self, max_length=100):
        self.dim = max_length
        self.pmf_matrix = np.zeros((max_length, max_length))
        self.pmf_matrix[0, 0] = 1

    def upstream_arrival(self, arrival_prob):
        new_matrix = np.zeros((self.dim, self.dim))
        new_matrix[:, 0] = self.pmf_matrix[:, 0] * (1 - arrival_prob)
        for _i in range(self.dim - 1):
            new_matrix[:, _i + 1] = self.pmf_matrix[:, _i] * arrival_prob +\
                                    self.pmf_matrix[:, _i + 1] * (1 - arrival_prob)
        self.pmf_matrix = new_matrix

    def internal_flow(self, departure_prob):
        new_matrix = np.zeros((self.dim, self.dim))
        new_matrix[0, :] = self.pmf_matrix[0, :] * (1 - departure_prob)
        for _i in range(self.dim - 1):
            for _j in range(self.dim - 1):
                if _j == 0:
                    new_matrix[_i + 1, _j] = self.pmf_matrix[_i + 1, _j] + \
                                             self.pmf_matrix[_i, _j + 1] * departure_prob
                else:
                    new_matrix[_i + 1, _j] = self.pmf_matrix[_i + 1, _j] * (1 - departure_prob) + \
                                             self.pmf_matrix[_i, _j + 1] * departure_prob
        new_matrix[0, 0] = self.pmf_matrix[0, 0]
        self.pmf_matrix = new_matrix

    def downstream_departure(self, departure_prob):
        new_matrix = np.zeros((self.dim, self.dim))
        new_matrix[0, :] = self.pmf_matrix[0, :] + self.pmf_matrix[1, :] * departure_prob
        for _i in range(self.dim - 2):
            new_matrix[_i + 1, :] = self.pmf_matrix[_i + 1, :] * (1 - departure_prob) +\
                                    self.pmf_matrix[_i + 2, :] * departure_prob
        self.pmf_matrix = new_matrix


def main():
    upstream_arrival_probability_list = [0.1, 0.2, 0.4, 0.8, 0.7, 0.6, 0.4, 0.2]
    # upstream_departure_probability_list = [0.1, 0.2, 0.3, 0.9, 0.9, 0.9, 0.8, 0.6]
    upstream_departure_probability_list = [0, 0, 0, 1, 1, 1, 1, 1]
    downstream_departure_probability_list = [0.9, 0.9, 0.9, 0.8, 0.6, 0.1, 0.2, 0.3]

    total_arrival = np.sum(upstream_arrival_probability_list)
    total_departure = np.sum(upstream_departure_probability_list)
    print("Total arrival", total_arrival,
          "total departure", total_departure,
          "-> valid", total_departure > total_arrival)

    simulation_steps = 2000
    upstream_queue = SingleQueue(max_length=30)
    downstream_queue = SingleQueue(max_length=30)

    joint_queue = JointQueue(max_length=30)
    independent_q_list = None
    dependent_q_list = None
    for i_t in range(simulation_steps):
        _t = i_t % len(upstream_arrival_probability_list)
        if _t == 0:
            print("========================================")
            pass
        arrival_prob = upstream_arrival_probability_list[_t]
        upstream_departure_prob = upstream_departure_probability_list[_t]
        downstream_departure_prob = downstream_departure_probability_list[_t]

        upstream_queue.arrival_step(arrival_prob)
        actual_departure = upstream_queue.departure_step(upstream_departure_prob)
        downstream_queue.arrival_step(actual_departure)
        downstream_queue.departure_step(downstream_departure_prob)
        # print(downstream_queue.pmf_list, sum(downstream_queue.pmf_list))

        joint_queue.upstream_arrival(arrival_prob)
        # print("========================")
        # print(joint_queue.pmf_matrix, np.sum(joint_queue.pmf_matrix))
        joint_queue.internal_flow(upstream_departure_prob)
        # print(joint_queue.pmf_matrix, np.sum(joint_queue.pmf_matrix))
        joint_queue.downstream_departure(downstream_departure_prob)
        # print(joint_queue.pmf_matrix, np.sum(joint_queue.pmf_matrix))
        # print(np.sum(joint_queue.pmf_matrix, 0))
        departure_marginal_dis = np.sum(joint_queue.pmf_matrix, 1)
        # print(departure_marginal_dis, sum(departure_marginal_dis))

        diff = np.array(downstream_queue.pmf_list) - departure_marginal_dis
        diff_metric = np.sum(np.abs(diff))
        # print("Difference", diff)
        print("Diff metric", diff_metric)
        # exit()
        independent_q_list = downstream_queue.pmf_list
        dependent_q_list = departure_marginal_dis

    plt.figure()
    plt.plot(independent_q_list[:10], label='approximated')
    plt.plot(dependent_q_list[:10], label='rigorous')
    plt.grid()
    plt.xlabel("Number of vehicles")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
