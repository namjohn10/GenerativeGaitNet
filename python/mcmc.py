import numpy as np

# target_dist (target_distribution) f : x(state) -> double(probability) 
# proposed_dist (proposed_distribution) f : x(current_state) -> x'(new_state) 
# for sampling, object(MetropolisHasting).get_sample f : int(size) -> list(sampled values)
 
class MetropolisHasting:
    def __init__(self, size, min_v, max_v, target_dist, proposed_dist):
        if min_v.size != size or max_v.size != size:
            raise ValueError("[MetropolisHasting] Min != Max != Size")
        self.min_v = min_v
        self.max_v = max_v
        self.cur_x = np.ndarray(size)
        self.size = size
        self.target_dist = target_dist
        self.proposed_dist = proposed_dist
        self.result = []
        self.reset()

    def reset(self):
        self.cur_x = self.proposed_dist(self.cur_x, self.min_v, self.max_v)
        # np.array([np.random.uniform(self.min_v[i], self.max_v[i]) for i in range(self.size)])
        # for _ in range(1000):
            # self.sample()
        self.result = []

    def compute_alpha(self, x_new, x_cur):
        # print('[DEBUG] ', self.target_dist(x_new) ,self.target_dist(x_cur) )
        v_cur = self.target_dist(x_cur)
        v_new = self.target_dist(x_new)
        if v_cur == 0:
            return 1.0
        # ratio = (self.target_dist(x_new)/self.target_dist(x_cur))
        return max(min(1.0, v_new / v_cur), 0)
    
    def sample(self):
        x_new = self.proposed_dist(self.cur_x, self.min_v, self.max_v)
        alpha = self.compute_alpha(x_new, self.cur_x)

        if np.random.rand() <= alpha:
            self.cur_x = x_new
        
        self.result.append(self.cur_x)

    def get_sample(self, iter):
        for _ in range(iter-1):
            self.sample()
        return self.result

#=============================Validation Test===========================
# min_v = np.ndarray(1)
# min_v[0] = -1

# max_v = np.ndarray(1)
# max_v[0] = 2


# # Target Distribution (x)
# def target_dist(x):
#     x = x * 2
#     if 0.25 < x and x < 0.5:
#         return 0
#     else:
#         return -x * x * x + x*x + 3*x + 10

# # Proposed Distribution (x)
# def proposed_dist(x, min_v, max_v):
#     size = x.size 
#     value = np.array([np.random.uniform(min_v[i], max_v[i]) for i in range(size)])
#     return value

# MH_test = MetropolisHasting(1, min_v, max_v, target_dist, proposed_dist)
# result = MH_test.get_sample(50000)

# # round_result = [round(result[i][0],2) for i in range(len(result))]
# resolution = 100.
# x = np.arange(min_v[0], max_v[0], 1.0 / resolution)
# y = np.zeros(x.size)

# # for i in range(len(x)):
# #     y[i] = target_dist(x[i])

# for i in range(len(result)):
#     y[int((result[i][0] - min_v[0])/(1.0 / resolution))] += 1

# import matplotlib.pyplot as plt   
# plt.plot(x, y)
# plt.show()
#=======================================================================
