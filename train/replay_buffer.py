
# Prioritized Replay Buffer

import numpy as np

class SumTree(object):
    # 建立tree和data。
    # 因为SumTree有特殊的数据结构
    # 所以两者都能用一个一维np.array来存储
    
    def __init__(self, capacity):
        self.data_pointer = 0
        self.data_count = 0
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    #当有新的sample时，添加进tree和data
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

        if self.data_count < self.capacity:
            self.data_count += 1

    #当sample被train,有了新的TD-error,就在tree中更新
    def update(self, tree_idx, p):
        assert p>=0
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    #根据选取的v点抽取样本
    def get(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class ReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 1e-6  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1e6  # clipped abs error

    #建立SumTree和各种参数
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    @property
    def data_count(self):
        return self.tree.data_count

    #存储数据，更新SumTree
    def push(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    #抽取sample
    def sample(self, n):
        if self.tree.data_count == 0: return [],[],[]
        if n < 0: n = self.tree.data_count
        b_idx = []
        b_memory = []
        ISWeights = []
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_priority = np.min(self.tree.tree[-self.tree.capacity:])
        if min_priority == 0:
            min_priority = np.min([x for x in self.tree.tree[-self.tree.capacity:] if x > 0])
        min_prob = min_priority / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i + (self.epsilon if i==0 else 0), pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get(v)
            prob = p / self.tree.total_p
            b_idx += [ idx ]
            for x in range(len(data)):
                d = data[x]
                if i == 0:
                    b_memory += [[d]]
                else:
                    b_memory[x] += [d]
            ISWeights += [ np.power(prob/min_prob, -self.beta) ]
        return b_idx, b_memory, ISWeights

    #train完被抽取的samples后更新在tree中的sample的priority
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def remove(self, tree_idx):
        for ti in tree_idx:
            self.tree.update(ti, 0)
            self.tree.data_count -= 1

if __name__ == '__main__':
    tree = SumTree(3)
    print("empty", tree.get(1))
    tree.add(0.1,(1,3))
    tree.add(2.5,(2,4))
    tree.add(0.2,(3,5))
    print(tree.tree)
    tree.add(4,(4,6))
    tree.add(6.9,(6,11))
    tree.add(6.1,(6,12))
    print(tree.tree)
    tree.add(3,(3,13))
    tree.add(0.2,(3,17))

    print(tree.tree)
    print(tree.data)
    for i in range(0,int(tree.tree[1])+1):
        print(i, tree.get(i))  # (8, 4.0, 6)

    cc={}
    pri_seg = tree.total_p / 10000
    for i in range(10000):
        a, b = pri_seg * i, pri_seg * (i + 1)
        v = np.random.uniform(a, b)
        idx, p, data = tree.get(v)
        cc[p] = (cc[p] if p in cc else 0) + 1
    print(cc)

    mem = ReplayBuffer(10)
    mem.push(('a'))
    mem.push(('b'))
    mem.push(('c'))
    mem.push(('d'))
    print(mem.tree.tree)
    print(mem.tree.data)
    print(list(mem.sample(5)))
    mem.remove([9,10,11,12])
    mem.push(('e'))
    mem.push(('f'))
    print(list(mem.sample(5)))
