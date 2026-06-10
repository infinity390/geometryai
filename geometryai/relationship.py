import copy
from mathai import simplify, fraction, dowhile, tree_form
def merge_category(cat, mergefx):
    used = [False] * len(cat)
    out = []
    for i in range(len(cat)):
        if used[i]:
            continue
        merged = list(cat[i])
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(cat)):
                if used[j]:
                    continue
                if mergefx(merged, cat[j]):
                    merged.extend(cat[j])
                    used[j] = True
                    changed = True
        seen = set()
        clean = []
        for x in merged:
            if x not in seen:
                seen.add(x)
                clean.append(x)
        out.append(clean)
    out = [x for x in out if len(x) != 0]
    return out
class Logic3D:
    def __init__(self, data=None):
        if data is None:
            self.data = []
        else:
            self.data = data
    def clean(self):
        self.data = [x for x in self.data if len(x) != 0]
        return self
    def solve(self):
        r = []
        while True:
            self.data = [list(sorted([tuple(sorted(item2)) for item2 in item])) for item in self.data]
            r.append(tuple([tuple(item) for item in self.data]))
            if r[-1] in r[:-1]:
                break
            self.merge()
            self.data = [list(sorted([list(sorted(item2)) for item2 in item])) for item in self.data]
            self.substitute()
        self.data = [list(sorted([list(sorted(item2)) for item2 in item])) for item in self.data]
        return self
    def logic2d(self):
        lst = []
        for item in self.data:
            lst2 = []
            for item2 in item:
                if len(item2) == 1:
                    lst2.append(item2[0])
            lst.append(lst2)
        return lst
    def merge(self):
        self.data = merge_category(self.data, lambda a, b: set(a) & set(b))
    def substitute(self):
        data = self.data
        lst = []
        for item in data:
            for item2 in item:
                if len(item2) == 1:
                    for item3 in item:
                        if item2 != item3:
                            lst.append(tuple((item2[0], tuple(item3))))
        for item in lst:
            for index, item2 in enumerate(data):
                lst2 = []
                for item3 in item2:
                    n = item3.count(item[0])
                    if n != 0:
                        item3 = [x for x in item3 if x != item[0]]
                        for i in range(n):
                            tmp = item3+[item[0]]*i+list(item[1])*(n-i)
                            lst2.append(tmp.copy())
                data[index] += lst2
                data[index] = list(set([tuple(sorted(x)) for x in data[index]]))
                data[index] = [list(x) for x in data[index]]
        self.data = data
def ss(eq):
    return dowhile(eq, lambda x: fraction(simplify(x)))
def rref(matrix):
    rows, cols = len(matrix), len(matrix[0])
    lead = 0
    for r in range(rows):
        if lead >= cols:
            return matrix
        i = r
        while ss(matrix[i][lead]) == tree_form("d_0"):
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    return matrix
        matrix[i], matrix[r] = matrix[r], matrix[i]
        lv = matrix[r][lead]
        matrix[r] = [ss(m / lv) for m in matrix[r]]
        for i in range(rows):
            if i != r:
                lv = matrix[i][lead]
                matrix[i] = [ss(m - lv * n) for m, n in zip(matrix[i], matrix[r])]
        lead += 1
    return matrix

class Row:
    def __init__(self, data=None):
        self.data = data
    def val(self):
        count_zero = 0
        count_one = 0
        count_other = 0
        index = None
        zero = tree_form("d_0")
        one = tree_form("d_1")
        for i, item in enumerate(self.data[:-1]):
            if item == one:
                count_one += 1
                index = i
            elif item == zero:
                count_zero += 1
            else:
                count_other += 1
        if count_other != 0:
            return None
        if count_one == 1:
            return index, ss(-self.data[-1])
        return None
    def __add__(self, other):
        return Row([self.data[i]+other.data[i] for i in range(len(self.data))])
    def __neg__(self):
        return Row([-item for item in self.data])
    
def make_row_from_sum(key_list, sum_list):
    lst = []
    one = tree_form("d_1")
    zero = tree_form("d_0")
    for item in key_list:
        if item in sum_list:
            lst.append(one)
        else:
            lst.append(zero)
    lst.append(zero)
    return Row(lst)

def logic_obj_row(key_list, eq):
    lst = []
    for item in eq.data:
        base = -make_row_from_sum(key_list, item[0])
        for item2 in item[1:]:
            lst.append(base +  make_row_from_sum(key_list, item2))
    return lst

def val_sum_row(key_list, val_sum):
    lst = []
    for key, item in val_sum.items():
        base = make_row_from_sum(key_list, list(key))
        base.data[-1] = ss(-item)
        lst.append(base)
    return lst

def solve_relationship(key_list, eq, val_sum):
    eq = eq.solve()
    m = logic_obj_row(key_list, eq) + val_sum_row(key_list, val_sum)
    m = [item.data for item in m]
    if m == []:
        return eq, val_sum
    m = rref(m)
    m = [Row(item) for item in m]
    lst = {}
    for item in m:
        out = item.val()
        if out is not None:
            val_sum[tuple([key_list[out[0]]])] = out[1]
            if out[1] in lst.keys():
                lst[out[1]].append(key_list[out[0]])
            else:
                lst[out[1]] = [key_list[out[0]]]
    data = copy.deepcopy(eq.data)
    for key, item in lst.items():
        item = [[x] for x in item]
        data.append(item)
    eq = Logic3D(data).solve()
    return eq, val_sum
