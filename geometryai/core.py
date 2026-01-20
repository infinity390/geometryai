import itertools
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont
class Graph:
    def __init__(self, space):
        self.n = len(space.point_location)
        self.adj = {i: set() for i in range(self.n)}
        self._build(space.give_connect())
    def _build(self, edges):
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
    def all_cycles(self):
        cycles = []
        visited = [False] * self.n
        def dfs(start, current, parent, path):
            visited[current] = True
            path.append(current)
            for nxt in self.adj[current]:
                if nxt == parent:
                    continue
                if nxt == start and len(path) > 2:
                    cycles.append(path.copy())
                elif not visited[nxt]:
                    dfs(start, nxt, current, path)
            path.pop()
            visited[current] = False
        for v in range(self.n):
            dfs(v, v, -1, [])
        return cycles
    def _canonical_cycle(self, cycle):
        cycle = cycle[:-1] if cycle[0] == cycle[-1] else cycle
        n = len(cycle)
        rotations = []
        for i in range(n):
            r = cycle[i:] + cycle[:i]
            rotations.append(tuple(r))
            rotations.append(tuple(reversed(r)))
        return min(rotations)
    def simple_cycles(self):
        raw = self.all_cycles()
        unique = set()
        for cycle in raw:
            if len(set(cycle)) != len(cycle):
                continue
            unique.add(self._canonical_cycle(cycle))
        return [list(c) for c in unique]
    def consecutive_triplets_at(self):
        triplets = []
        for v in self.adj.keys():
            nbrs = list(self.adj[v])
            if len(nbrs) >=2:
                for i in range(len(nbrs)):
                    for j in range(len(nbrs)):
                        if i != j:
                            triplets.append((nbrs[i], v, nbrs[j]))
        return triplets
def draw_geometry(points, edges, size, margin):
    pts = [(float(x), float(y)) for x, y in points]
    xs = [x for x, y in pts]
    ys = [y for x, y in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x or 1
    height = max_y - min_y or 1
    if size is None:
        size = 300
    if margin is None:
        margin = 40
    scale = min(
        (size - 2 * margin) / width,
        (size - 2 * margin) / height
    )
    def transform(x, y):
        px = margin + (x - min_x) * scale
        py = size - (margin + (y - min_y) * scale)
        return px, py

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for i, j in edges:
        p1 = transform(*pts[i])
        p2 = transform(*pts[j])
        draw.line([p1, p2], fill="black", width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    r = 4
    label_offset = (6, -6)
    for idx, (x, y) in enumerate(pts):
        px, py = transform(x, y)
        draw.ellipse((px-r, py-r, px+r, py+r), fill="red")
        label = chr(ord("A") + idx)
        draw.text(
            (px + label_offset[0], py + label_offset[1]),
            label,
            fill="blue",
            font=font
        )
    img.save("output.png")
    return img
def merge_category(cat, mergefx):
    n = len(cat)
    used = [False] * n
    out = []
    for i in range(n):
        if used[i]:
            continue
        merged = []
        for j in range(i, n):
            if not used[j] and mergefx(cat[i], cat[j]):
                merged += cat[j]
                used[j] = True
        out.append(merged)
    return [list(set(item)) for item in out]

def are_collinear(points):
    """
    Returns True if all points are collinear, False otherwise.
    points: list of (x, y), x and y can be Fraction or int
    """
    n = len(points)
    if n <= 2:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for i in range(2, n):
        xi, yi = points[i]
        if (xi - x0) * dy != (yi - y0) * dx:
            return False
    return True
def intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det
    return (x, y)

def point_sort(point):
    if isinstance(point, str):
        return ord(point)-ord("A")
    return point
def line_sort(line):
    if isinstance(line, str):
        return tuple(sorted([ord(item)-ord("A") for item in line]))
    return tuple(sorted(list(line)))
class Space:
    def __init__(self):
        self.point_location = []
        self.line_info = []
        self.angle_list = {}
        self.command = []
        self.line = []
        self.ray = []
        self.graph = None
        self.line_eq = []
        self.angle_eq = []
        self.tri_eq = []
        self.perpendicular_angle = []
        self.perpendicular = []
    def standard_angle(self, angle):    
        if isinstance(angle, str):
            angle = tuple([ord(item)-ord("A") for item in angle])
        if angle[0] > angle[2]:
            angle = (angle[2],angle[1],angle[0])
        if isinstance(angle, list):
            angle = tuple(angle)
        for key in self.angle_list.keys():
            if key == angle or angle in self.angle_list[key]:
                return key
        return None
    def perpen_angle(self):
        self.perpendicular = [[list(item2) for item2 in item] for item in self.perpendicular]
        for item in itertools.combinations(self.line_info, 2):
            item = list(item)            
            if all(not self.straight_line(list(set(item[0]+item2[0]))) or not self.straight_line(list(set(item[1]+item2[1]))) for item2 in self.perpendicular)\
               and all(not self.straight_line(list(set(item[0]+item2[1]))) or not self.straight_line(list(set(item[1]+item2[0]))) for item2 in self.perpendicular):
                continue
            if len(set(item[0])&set(item[1]))==1:
                c = list(set(item[0])&set(item[1]))[0]
                a = item[0].index(c)
                b = item[1].index(c)
                m, n = [], []
                for i in range(2):
                    d = [a,b][i]
                    if d>0:
                        [m,n][i].append(item[i][d-1])
                    if d<len(item[i])-1:
                        [m,n][i].append(item[i][d+1])
                for item2 in itertools.product(m,n):
                    angle = space.standard_angle((item2[0],c,item2[1]))
                    if angle not in self.perpendicular_angle:
                        self.perpendicular_angle.append(angle)
        space.angle_eq.append(self.perpendicular_angle)
        space.angle_eq = merge_category(space.angle_eq, default_merge)
    def straight_line(self, point_list):
        return are_collinear([self.point_location[x] for x in point_list])
    def sort_collinear(self, point_list):
        p = min([self.point_location[x][1] for x in point_list])
        p2 = [x for x in point_list if self.point_location[x][1] == p]
        p3 = list(sorted(p2, key=lambda x: self.point_location[x][0]))[0]
        m, n = self.point_location[p3]
        return list(sorted(point_list, key=lambda x: (self.point_location[x][0]-m)**2 + (self.point_location[x][1]-n)**2))
    def calc_angle_list(self):
        lst = self.graph.consecutive_triplets_at()
        lst = list(set([item if item[0]<item[2] else (item[2],item[1],item[0]) for item in lst]))
        lst = [item for item in lst if not self.straight_line(list(item))]
        for item2 in lst:
            self.angle_list[item2] = []
            for item in itertools.permutations(self.line_info, 2):
                if item2[0] in item[0] and item2[2] in item[1] and item2[1] in item[0] and item2[1] in item[1]:
                    h = []
                    for i in range(2):
                        m = item[i][:item[i].index(item2[1])]
                        n = item[i][item[i].index(item2[1])+1:]
                        if item2[0] in m or item2[2] in m:
                            h.append(m)
                        elif item2[0] in n or item2[2] in n:
                            h.append(n)
                    for item3 in itertools.product(*h):
                        x = (item3[0], item2[1], item3[1])
                        y = (item3[1], item2[1], item3[0])
                        self.angle_list[item2] += [x,y]
    def give_connect(self):
        out = []
        for item in self.line_info:
            for i in range(len(item)-1):
                out.append(item[i:i+2])
        return out
    def line_eq_fx(self, line1, line2):
        line1 = line_sort(line1)
        line2 = line_sort(line2)
        if line1 == line2:
            return True
        for item in self.line_eq:
            if line1 in item and line2 in item:
                return True
        return False
    def angle_eq_fx(self, angle1, angle2):
        angle1 = self.standard_angle(angle1)
        angle2 = self.standard_angle(angle2)
        if angle1 == angle2:
            return True
        for item in self.angle_eq:
            if angle1 in item and angle2 in item:
                return True
        return False
    def valid_line(self, line):
        line = line_sort(line)
        return any((line[0] in item and line[1] in item) for item in self.line_info)
    def show_diagram(self, size):
        out = draw_geometry(self.point_location, self.give_connect(), size, None)
        try:
            from IPython.display import display
            display(out)
            out.show()
        except:
            print("error displaying image")
    def calc_line_info(self):
        line = [self.line_info[x] for x in self.line]
        ray = [(self.line_info[x[0]], x[1]) for x in self.ray]
        self.line = []
        self.ray = []
        cat = []
        for index in range(2):
            for item in itertools.combinations(list(range(len(self.point_location))), 3):
                if self.straight_line(list(item)):
                    cat.append(list(item))
            for item in self.line_info:
                if len(item) == 2 and all(item[0] not in item2 or item[1] not in item2 for item2 in cat):
                    cat.append(list(item))
            def mergefx(a, b):
                return self.straight_line(list(set(a+b)))
            cat = merge_category(cat, mergefx)
            p = []
            
            for item in itertools.combinations(cat, 2):
                p2 = intersection(*[self.point_location[item2] for item2 in item[0][:2]+item[1][:2]])
                p.append(p2)
            p = list(set(p))
            if self.command != [] and index == 0:
                for i in range(len(self.command)-1,-1,-1):
                    for item in p:
                        if any(item2[0]==item[0] and item2[1]==item[1] for item2 in self.point_location):
                            continue
                        self.point_location.append(item)
                        lst = [self.command[i][0], len(self.point_location)-1, self.command[i][1]]
                        if self.straight_line(lst) and (any(lst[0] in item2 and lst[1] in item2 for item2 in self.line) or self.sort_collinear(lst)[1] == lst[1]):
                            pass
                        else:
                            self.point_location.pop(-1)
                cat = []
            else:
                break
        self.line_info = [self.sort_collinear(item) for item in cat]
        for item in line:
            for i in range(len(self.line_info)):
                if self.straight_line(list(self.line_info[i])+item):
                    self.line.append(i)
        for item in ray:
            for i in range(len(self.line_info)):
                if self.straight_line(list(self.line_info[i])+item[0]):
                    self.ray.append((i, item[1]))
        self.graph = Graph(self)
def default_merge(a, b):
    return (set(a)&set(b)) != {}
space = None
def draw_triangle():
    global space
    space = Space()
    space.point_location = [
        (Fraction(0), Fraction(0)),
        (Fraction(4), Fraction(1)),
        (Fraction(1), Fraction(3)),
    ]
    space.line_info = [[0,1],[1,2],[2,0]]
def given_equal_line(line1, line2):
    global space
    line1 = line_sort(line1)
    line2 = line_sort(line2)
    space.line_eq.append([line1, line2])
    space.line_eq = merge_category(space.line_eq, default_merge)
def cpct():
    global space
    for item in space.tri_eq:
        for item2 in itertools.combinations(item, 2):
            m2 = list(zip(*item2))
            for item3 in itertools.permutations(m2):
                angle1, angle2 = (item3[0][0], item3[1][0], item3[2][0]), (item3[0][1], item3[1][1], item3[2][1])
                angle1, angle2 = space.standard_angle(angle1), space.standard_angle(angle2)
                
                if angle1 is None or angle2 is None or angle1 == angle2:
                    continue
                space.angle_eq.append([angle1, angle2])
            for item3 in itertools.combinations(m2, 2):
                line1, line2 = item3
                line1, line2 = line_sort(line1), line_sort(line2)
                if not space.valid_line(line1) or not space.valid_line(line2) or line1 == line2:
                    continue
                space.line_eq.append([line1, line2])
    space.line_eq = merge_category(space.line_eq, default_merge)
    space.angle_eq = merge_category(space.angle_eq, default_merge)
def sss_rule(a1, a2, a3, b1, b2, b3):
    global space
    a1, a2, a3, b1, b2, b3 = [[item] for item in [a1, a2, a3, b1, b2, b3]]
    line = [
        line_sort(a1 + a2),
        line_sort(b1 + b2),
        line_sort(a2 + a3),
        line_sort(b2 + b3),
        line_sort(a1 + a3),
        line_sort(b1 + b3),
    ]
    
    for item in line:
        if not space.valid_line(item):
            return False
        
    return (
        space.line_eq_fx(line[0], line[1])
        and space.line_eq_fx(line[2], line[3])
        and space.line_eq_fx(line[4], line[5])
    )
def rhs_rule(a1, a2, a3, b1, b2, b3):
    global space
    a1, a2, a3, b1, b2, b3 = [[item] for item in [a1, a2, a3, b1, b2, b3]]
    line = [
        line_sort(a1 + a2),
        line_sort(b1 + b2),
        line_sort(a1 + a3),
        line_sort(b1 + b3),
    ]
    angle = [space.standard_angle(a1 + a2 + a3), space.standard_angle(b1 + b2 + b3)]

    for item in line:
        if not space.valid_line(item):
            return False

    for item in angle:
        if item is None:
            return False
        
    return (
        space.line_eq_fx(line[0], line[1])
        and space.angle_eq_fx(angle[0], angle[1])
        and space.line_eq_fx(line[2], line[3])
        and angle[0] in space.perpendicular_angle
    )
def tri_sort(tri):
    return tuple([ord(item)-ord("A") for item in tri])
def check_equal_angle(a, b):
    global space
    return space.angle_eq_fx(a, b)
def check_equal_line(a, b):
    global space
    return space.line_eq_fx(a, b)
def prove_congruent_triangle(tri1, tri2=None):
    global space
    if tri2 is None:
        tri2 = tri1
    list1 = list(itertools.permutations(list(tri_sort(tri1))))
    list2 = list(itertools.permutations(list(tri_sort(tri2))))
    for x in list1:
        for y in list2:
            a = list(x)
            b = list(y)
            for item in [a+b,b+a]:
                for rule in [sss_rule, rhs_rule]:
                    out = rule(*item)
                    if out:
                        space.tri_eq.append([x, y])
    space.tri_eq = merge_category(space.tri_eq, default_merge)
def process():
    global space
    space.calc_line_info()
    space.calc_angle_list()
    space.perpen_angle()
def split_line(line, p=None):
    global space
    line = line_sort(line)
    a, b = space.point_location[line[0]], space.point_location[line[1]]
    px = (a[0]+b[0])/2
    py = (a[1]+b[1])/2
    if p is not None:
        px, py = p
    space.point_location.append((px, py))
    r = len(space.point_location)-1
    for i in range(len(space.line_info)-1,-1,-1):
        if line[0] in space.line_info[i] and line[1] in space.line_info[i]:
            
            space.line_info[i] = space.line_info[i][:min(line)]+[len(space.point_location)-1]+space.line_info[i][max(line):]
            
    return r
def extended_line(line):
    global space
    line = line_sort(line)
    for i in range(len(space.line_info)):
        if line[0] in space.line_info[i] and line[1] in space.line_info[i]:
            self.line.append(i)
def join(line):
    global space
    line = line_sort(line)
    space.line_info.append(line)
    space.command.append(line)
def foot_of_perpendicular(P, A, B):
    x0, y0 = P
    x1, y1 = A
    x2, y2 = B
    dx = x2 - x1
    dy = y2 - y1
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
    x = x1 + t * dx
    y = y1 + t * dy
    return (x, y)
def draw_perpendicular(point, line):
    global space
    point = point_sort(point)
    line = line_sort(line)
    out = foot_of_perpendicular(space.point_location[point], space.point_location[line[0]], space.point_location[line[1]])
    out2 = split_line(line, out)
    m = line_sort((point, out2))
    join(m)
    space.perpendicular.append([m,line])
def show(size=None):
    global space
    space.show_diagram(size)
def norm_angle(angle):
    global space
    return space.standard_angle(angle)
