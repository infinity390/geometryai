import copy
import itertools
import math
from fractions import Fraction
from collections import Counter
from mathai import parse, frac, TreeNode, contain2, dowhile, compute, str_form, transform_dfs,\
     factor_generation, product, string_equation, tree_form, summation, flatten_tree, simplify
def frac_to_tree_2(eq):
    if isinstance(eq, int):
        eq = Fraction(eq)
    if eq.numerator == 0:
        return tree_form("d_0")
    if eq.denominator == 1:
        return tree_form(f"d_{eq.numerator}")
    return TreeNode("f_div", [tree_form(f"d_{eq.numerator}"), tree_form(f"d_{eq.denominator}")])
def perfect_sqrt(n):
    if isinstance(n, Fraction):
        if n.denominator == 1:
            n = n.numerator
        else:
            return None
    if n < 0:
        return None
    r = math.isqrt(n)
    if r * r == n:
        return Fraction(r)
    return None
def simplify_sqrt(a):
    b = 1
    c = a
    i = 2
    while i*i <= c:
        while c % (i*i) == 0:
            c //= i*i
            b *= i
        i += 1
    return Fraction(b), Fraction(c)
def coeff(eq):
    out = frac(eq)
    if out is not None:
        return out
    lst = factor_generation(eq)
    lst2 = [frac(item) for item in lst if frac(item) is not None]
    lst3 = product([item for item in lst if frac(item) is None])
    k = Fraction(1)
    for item in lst2:
        k = k *item
    return k, lst3
def nested_radical(g):
    out = {}
    lst = []
    for key, item in g.dic.items():
        if isinstance(key, G):
            out2 = formula(key.simplify())
            key, cmd = out2
            if cmd == "denest":
                lst.append(G(key.eq()))
            else:
                out[key] = item
        else:
            out[key] = item
    tmp = G(out, g.const).simplify()
    for item in lst:
        tmp = tmp + item
    return tmp, "intact"
def formula(g):
    if len(g.dic.keys()) != 1:
        return g, "intact"
    c = list(g.dic.keys())[0]
    if isinstance(c, G):
        c = nested_radical(c)[0].simplify()
    if isinstance(c, G):
        return G({c:g.dic[c]},g.const).simplify(), "intact"
    a = g.const
    b = g.dic[c]
    tmp = a**2 - b**2*c
    if not isinstance(tmp, Fraction):
        if tmp.denominator == 1:
            tmp = tmp.numerator
        else:
            return G({c:b},a).simplify(), "intact"
    out = perfect_sqrt(tmp)
    if out is None:
        return G({c:b},a).simplify(), "intact"
    x = (a + out)/Fraction(2)
    y = (a - out)/Fraction(2)
    new_dic = {x:Fraction(1), y:Fraction(1)}
    return G(new_dic).simplify(), "denest"
def eq(self):
    return TreeNode(self.name, [child.eq() for child in self.children])
TreeNode.eq = eq
class G:
    def __init__(self, dic, const=Fraction(0)):
        if isinstance(const, int):
            const = Fraction(const)
        self.const = const
        self.dic = {}
        if isinstance(dic, TreeNode):
            out = frac(dic)
            if out is not None:
                self.const += out
                return
            if dic.name != "f_add":
                dic = TreeNode("f_add", [dic])
            if dic.name == "f_add":
                ch = dic.children
                for child in ch:
                    out = coeff(helper_2(child))
                    if isinstance(out, Fraction):
                        self.const += out
                    else:
                        n = None
                        if out[1].name == "f_sqrt" and out[1].children[0].name.startswith("d_"):
                            n = int(out[1].children[0].name[2:])
                            n = Fraction(n)
                        else:
                            n = G(out[1].children[0])
                        if n in self.dic.keys():
                           self.dic[n] += out[0]
                        else:
                            self.dic[n] = out[0]
        else:
            for key, item in dic.items():
                if isinstance(key, int):
                    key = Fraction(key)
                if isinstance(item, int):
                    item = Fraction(item)
                self.dic[key] = item
    def simplify(self):
        new_dic = {}
        for key, item in self.dic.items():
            if isinstance(key, G) and key.dic == {}:
                new_dic[key.const] = item
            else:
                new_dic[key] = item
        self.dic = new_dic
        while True:
            out = None
            rm = None
            for key in self.dic.keys():
                if isinstance(key, Fraction):
                    a, b = simplify_sqrt(key)
                    if a != 1:
                        out = G({b:a*self.dic[key]})
                        rm = key
                        break
            if out is None:
                break
            del self.dic[rm]
            self = G(self.dic, self.const) + out
        lst = [key for key, item in self.dic.items() if item == 0 or (not isinstance(key, G) and key == Fraction(1))]
        for key, item in self.dic.items():
            if not isinstance(key, G) and key == Fraction(1):
                self.const += item
        for item in lst:
            del self.dic[item]
        return self
    def __add__(self, other):
        out = {}
        for key in list(set(list(self.dic.keys())+list(other.dic.keys()))):
            out[key] = Fraction(0)
            if key in self.dic.keys():
                out[key] += self.dic[key]
            if key in other.dic.keys():
                out[key] += other.dic[key]
        return G(out, self.const + other.const)
    def __mul__(self, other):
        out = {}
        for a, ca in list(self.dic.items()) + [(Fraction(1), self.const)]:
            for b, cb in list(other.dic.items()) + [(Fraction(1), other.const)]:
                if isinstance(a, Fraction) and isinstance(b, Fraction):
                    pass
                else:
                    if isinstance(a, Fraction):
                        a = G({}, a)
                    if isinstance(b, Fraction):
                        b = G({}, b)
                c = a * b
                if isinstance(c, G):
                    c = c.simplify()
                    if c.dic == {}:
                        c = c.const
                out[c] = out.get(c, Fraction(0)) + ca * cb
        return G(out)
    def __truediv__(self, other):
        other = nested_radical(other)[0]
        self = nested_radical(self)[0]
        if self.const == Fraction(0) and self.dic == {}:
            return self
        if any(isinstance(item, G) for item in other.dic.keys()):
            return TreeNode("f_div", [self.eq(), other.eq()])
        n = len(list(other.dic.keys()))
        lst = []
        if n != 0:
            for item in itertools.product([Fraction(1),Fraction(-1)], repeat=n):
                item = list(item)
                new_g = G(other.dic, other.const)
                key_list = list(new_g.dic.keys())
                for i in range(len(item)):
                    new_g.dic[key_list[i]] *= item[i]
                lst.append(new_g)
            lst.remove(other)
            mul = lst[0]
            for item in lst[1:]:
                mul = mul * item
            norm = mul * other
            norm = nested_radical(norm)[0]
            ans = mul * self
        else:
            norm = G({}, other.const)
            ans = self
        ans = nested_radical(ans)[0]
        for key, item in ans.dic.items():
            ans.dic[key] = item/norm.const
        ans.const = ans.const / norm.const
        return ans
    def __eq__(self, other):
        a, b = map(lambda x: str_form(x.eq()), [self,other])
        return a == b
    def __hash__(self):
        return hash(str_form(self.eq()))
    def __repr__(self):
        return str(self.eq())
    def eq(self):
        equation = []
        for key, item in self.dic.items():
            if isinstance(key, Fraction):
                out = frac_to_tree_2(key).fx("sqrt")
            else:
                out = key.eq().fx("sqrt")
            if item == 1:
                equation.append(out)
            elif item != 0:
                equation.append(frac_to_tree_2(item)*out)
        final = None
        if self.const == 0:
            final = summation(equation)
        else:
            final = summation(equation + [frac_to_tree_2(self.const)])
        return final

def helper_2_h(eq):
    if eq.name == "f_sqrt":
        return eq.children[0] ** (tree_form("d_2")**tree_form("d_-1"))
    if eq.name == "f_div":
        return eq.children[0] * eq.children[1]**tree_form("d_-1")
    return eq
def helper_3_h(eq):
    if eq.name == "f_sub":
        return eq.children[0] - eq.children[1]
    if eq.name == "f_neg":
        return -eq.children[0]
    return eq
def helper_2(eq):
    return transform_dfs(eq, helper_2_h)
def helper_3(eq):
    return transform_dfs(eq, helper_3_h)
def valid(eq):
    if eq.name == "f_div" and contain2(eq.children[1], "f_sqrt"):
        return False
    if contain2(eq, "f_mul"):
        return False
    for child in eq.children:
        if not valid(child):
            return False
    return True
def calc(eq):
    if isinstance(eq, TreeNode) and valid(eq):
        return G(eq)
    if isinstance(eq, TreeNode):
        if eq.name == "f_add":
            lst = [calc(item) for item in eq.children]
            lst2 = G({},Fraction(0))
            for i in range(len(lst)-1,-1,-1):
                if isinstance(lst[i], G):
                    lst2 = lst2 + lst[i]
                    lst.pop(i)
            if len(lst) > 0:
                return summation(lst) + lst2.eq()
            return lst2
        if eq.name == "f_mul":
            lst = [calc(item) for item in eq.children]
            lst2 = G({},Fraction(1))
            for i in range(len(lst)-1,-1,-1):
                if isinstance(lst[i], G):
                    lst2 = lst2 * lst[i]
                    lst.pop(i)
            if len(lst) > 0:
                return product(lst) * lst2.eq()
            return lst2
        if eq.name == "f_sqrt":
            out = calc(eq.children[0])
            return G({out:Fraction(1)}, 0)
        if eq.name == "f_pow" and eq.children[1] == 2:
            return calc(eq.children[0]) * calc(eq.children[0])
        if eq.name == "f_div":
            out = calc(eq.children[0])
            if isinstance(out, G) and out.dic == {} and out.const == 0:
                return G({})
            out2 = calc(eq.children[1])
            if isinstance(out, TreeNode) or isinstance(out2, TreeNode):
                if isinstance(out, G):
                    out = out.eq()
                if isinstance(out2, G):
                    out2 = out2.eq()
                return TreeNode("f_div", [out, out2])
            tmp = out/out2
            return tmp
        return None
    return eq
def solve_sqrt_h(eq):    
    x = calc(flatten_tree(helper_3(eq)))
    if x is None:
        return None
    if isinstance(x, G):
        x = nested_radical(x)[0].eq()
    return flatten_tree(x)
def dowhile2(eq, fx):
    if eq is None:
        return None
    lst = []
    while True:
        lst.append(copy.deepcopy(eq))
        eq2 = fx(eq)
        if eq2 is None:
            return None
        eq = copy.deepcopy(eq2)
        if eq in lst:
            return lst[-1]
def solve_sqrt(eq):
    fx = lambda x: transform_dfs(x, solve_sqrt_h)
    return dowhile(eq, lambda x: simplify(solve_sqrt_h(x), False))
def print_raw(eq):
    print(string_equation(str_form(eq)))
def frac_2(eq):
    return frac(helper_2(eq))
def simp(eq):
    if isinstance(eq, F):
        eq = eq.value
    if isinstance(eq, TreeNode):
        eq = solve_sqrt(eq)
        out = frac_2(eq)
        if out is None:
            return eq
        return out
    return eq
def get_value(x):
    if isinstance(x, F):
        return x.value
    if isinstance(x, int):
        return Fraction(x)
    return x
def promote_obj(x):
    if isinstance(x, F):
        if isinstance(x.value, TreeNode):
            return x.value
        return frac_to_tree_2(x.value)
    if isinstance(x, Fraction):
        return frac_to_tree_2(x)
    if isinstance(x, int):
        return frac_to_tree_2(Fraction(x))
    if isinstance(x, TreeNode):
        return x
    return x
def compare(a, b):
    a = get_value(a)
    b = get_value(b)
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        if a == b:
            return 0
        return 1 if a > b else -1
    vals = []
    for item in [a, b]:
        if isinstance(item, TreeNode):
            item = helper_2(item)
            item = compute(item)
        vals.append(item)
    if any(isinstance(x, float) for x in vals):
        vals = [
            float(x) if not isinstance(x, float) else x
            for x in vals
        ]
    if vals[0] == vals[1]:
        return 0
    return 1 if vals[0] > vals[1] else -1  
class F:
    def __init__(self, value=0):
        if isinstance(value, TreeNode):
            self.value = value
        elif isinstance(value, Fraction):
            self.value = value
        elif isinstance(value, int):
            self.value = Fraction(value)
        elif isinstance(value, F):
            self.value = value.value
        else:
            self.value = value
        if isinstance(self.value, TreeNode):
            self.value = simp(self.value)
    def promote(self, other, op):
        return F(
            TreeNode(
                op,
                [
                    promote_obj(self),
                    promote_obj(other)
                ]
            )
        )
    def fx(self, name):
        return F(
            TreeNode(
                "f_"+name,
                [
                    promote_obj(self)
                ]
            )
        )
    def __add__(self, other):
        if isinstance(self.value, TreeNode) or  isinstance(get_value(other), TreeNode):
            return self.promote(other, "f_add")
        return F(get_value(self) + get_value(other))
    def __sub__(self, other):
        if isinstance(self.value, TreeNode) or  isinstance(get_value(other), TreeNode):
            return self.promote(other, "f_sub")
        return F(get_value(self) - get_value(other))
    def __neg__(self):
        if isinstance(self.value, TreeNode):
            return F(TreeNode("f_neg",[promote_obj(self)]))
        return F(-self.value)
    def __mul__(self, other):
        if isinstance(get_value(self), TreeNode) or  isinstance(get_value(other), TreeNode):
            return self.promote(other, "f_mul")
        return F(get_value(self) * get_value(other))
    def __truediv__(self, other):
        if isinstance(get_value(self), TreeNode) or  isinstance(get_value(other), TreeNode):
            return self.promote(other, "f_div")
        return F(get_value(self) / get_value(other))
    def __pow__(self, other):
        if isinstance(get_value(self), TreeNode) or  isinstance(get_value(other), TreeNode):
            return self.promote(other, "f_pow")
        return F(get_value(self) ** get_value(other))
    def __hash__(self):
        if isinstance(get_value(self), Fraction):
            return hash(get_value(self))
        return hash(str_form(get_value(self)))
    def __lt__(self, other):
        return compare(self, other) == -1
    def __le__(self, other):
        return compare(self, other) in [-1, 0]    
    def __gt__(self, other):
        return compare(self, other) == 1
    def __ge__(self, other):
        return compare(self, other) in [1, 0]
    def __eq__(self, other):
        return compare(self, other) == 0
    def __float__(self):
        value = get_value(self)
        if isinstance(value, TreeNode):
            return compute(helper_2(value))
        return float(value)
    def __repr__(self):
        return str(self.value)
