from mathai import parse, frac, tree_form
from .relationship import solve_relationship, Logic3D, merge_category
from .coordinate import F
import copy
import itertools
from PIL import Image, ImageDraw, ImageFont
import math
import random
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

def draw_geometry(points, edges, circle, size=None, margin=None):
    if margin is None:
        margin = 25

    pts = [(float(x), float(y)) for x, y in points]

    xs = [x for x, y in pts]
    ys = [y for x, y in pts]

    for item in circle:
        cx, cy = map(float, points[item[0]])
        r = float(item[1])
        xs.extend([cx-r, cx+r])
        ys.extend([cy-r, cy+r])

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x or 1
    height = max_y - min_y or 1
    if size is None:
        size = int(max(width, height) * 100 + 2 * margin)
    scale = (size - 2*margin) / max(width, height)
    def transform(x,y):
        return (
            margin + (x-min_x)*scale,
            size - (margin + (y-min_y)*scale)
        )
    img = Image.new("RGB",(size,size),"white")
    draw = ImageDraw.Draw(img)
    for item in circle:
        cx,cy = transform(*map(float, points[item[0]]))
        radius = float(item[1])*scale
        draw.ellipse(
            (cx-radius, cy-radius, cx+radius, cy+radius),
            outline="black",
            width=2
        )

    for i,j in edges:
        draw.line(
            [transform(*pts[i]), transform(*pts[j])],
            fill="black",
            width=2
        )

    try:
        font = ImageFont.truetype("arial.ttf",22)
    except:
        font = ImageFont.load_default()

    for idx,(x,y) in enumerate(pts):
        px,py = transform(x,y)

        draw.ellipse(
            (px-4,py-4,px+4,py+4),
            fill="red"
        )

        draw.text(
            (px+6,py-6),
            chr(ord("A")+idx),
            fill="blue",
            font=font
        )

    img.save("output.png")
    return img

def are_collinear(points):
    n = len(points)
    if n <= 2:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for i in range(2, n):
        xi, yi = points[i]
        if not( (xi - x0) * dy == (yi - y0) * dx ):
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
        self.line_list = []
        self.command = []
        self.line = []
        self.ray = []
        self.graph = None
        self.line_eq = Logic3D()
        self.angle_eq = Logic3D()
        self.angle_val = {}
        self.tri_eq = Logic3D()
        self.tri_list = []
        self.parallel_eq = Logic3D()
        self.parallel_list = []
        self.perpendicular_angle = []
        self.perpendicular = []
        self.circle = []
        self.circle_arc = {}
    def point_on_circle(self, point, center, radius):
        x, y = self.point_location[point]
        ax, ay = self.point_location[center]
        return (x - ax)**2 + (y - ay)**2 == radius**2
    def update_arc(self):
        for i in range(len(self.circle)):
            for j in range(len(self.point_location)):
                if self.point_on_circle(j, self.circle[i][0], self.circle[i][1]):
                    if i in self.circle_arc.keys():
                        if j not in self.circle_arc[i]:
                            self.circle_arc[i].append(j)
                    else:
                        self.circle_arc[i] = [j]
        for item in self.circle:
            center = item[0]
            lst = []
            for item2 in self.circle_arc[center]:
                lst.append([line_sort([item2, center])])    
            space.line_eq.data.append(lst)
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
    def line_index(self, line):
        if isinstance(line, str):
            line = tuple([ord(item)-ord("A") for item in line])
        for index, item in enumerate(self.line_info):
            if line[0] in item and line[1] in item:
                return index
        return None
    def perpen_angle(self):
        self.perpendicular = [[list(item2) for item2 in item] for item in self.perpendicular]
        for item2 in self.perpendicular:
            item = list(map(lambda x: self.line_info[self.line_index(line_sort(x))], item2))
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
        space.angle_eq.data.append([[x] for x in self.perpendicular_angle])
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
        for item2 in itertools.combinations(lst, 2):
            if item2[0][1] == item2[1][1] and len(set([item2[0][0], item2[0][2]]) & set([item2[1][0], item2[1][2]]))==0:
                self.angle_eq.data.append([[self.standard_angle(list(item2[0]))], [self.standard_angle(list(item2[1]))]])
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
        for item in self.line_eq.logic2d():
            if line1 in item and line2 in item:
                return True
        return False
    def angle_eq_fx(self, angle1, angle2):
        angle1 = self.standard_angle(angle1)
        angle2 = self.standard_angle(angle2)
        if angle1 == angle2:
            return True
        for item in self.angle_eq.logic2d():
            if angle1 in item and angle2 in item:
                return True
        return False
    def valid_line(self, line):
        line = line_sort(line)
        if line[0] == line[1]:
            return False
        return any(line[0] in item and line[1] in item for item in self.line_info)
    def show_diagram(self, size):
        out = draw_geometry(self.point_location, self.give_connect(), self.circle, size, None)
        try:
            from IPython.display import display
            display(out)
            out.show()
        except:
            print("error displaying image")
    def calc_line_info(self):
        line = [self.line_info[x] for x in self.line]
        self.line = []
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
                if p2 is not None:
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
        self.graph = Graph(self)
space = None
def arc_split_midpoints(center, B, C, radius=None):

    ox, oy = center

    ux = B[0]-ox
    uy = B[1]-oy

    vx = C[0]-ox
    vy = C[1]-oy


    r = (ux*ux + uy*uy).fx("sqrt")

    if radius is None:
        radius = r


    ux /= r
    uy /= r

    vx /= r
    vy /= r


    wx = ux + vx
    wy = uy + vy


    lw = (wx*wx + wy*wy).fx("sqrt")

    if lw == 0:
        raise Exception("Diameter")


    wx /= lw
    wy /= lw


    p_plus = (
        ox + radius*wx,
        oy + radius*wy
    )

    p_minus = (
        ox - radius*wx,
        oy - radius*wy
    )


    # central angle
    dot = ux*vx + uy*vy


    # if angle is less than 180, + direction is minor
    if dot > -1:
        minor = p_plus
        major = p_minus
    else:
        minor = p_minus
        major = p_plus


    return minor, major
def arc_split(chord_a, chord_b, mode="minor"):
    global space
    if isinstance(chord_a, str):
        chord_a = ord(chord_a)-ord("A")
    if isinstance(chord_b, str):
        chord_b = ord(chord_b)-ord("A")
    center, radius = [item for index, item in enumerate(space.circle) if chord_a in space.circle_arc[index] and\
                      chord_b in space.circle_arc[index]][0]
    A = space.point_location[chord_a]
    B = space.point_location[chord_b]
    out = arc_split_midpoints(space.point_location[center], A, B)
    if mode == "minor":
        out = out[0]
    elif mode == "major":
        out = out[1]
    space.point_location.append(out)
    space.update_arc()
    join([len(space.point_location)-1, center])
    return None
def draw_circle():
    global space
    space = Space()
    radius = F(1)
    space.point_location = [
        (F(0), F(0)),
        (F(0), radius),
        (-radius, F(0))
    ]
    space.line_info = [[0,1],[0,2]]
    space.circle.append((0,radius))
    space.circle_arc = {0:[2,1]}
def draw_triangle():
    global space
    space = Space()
    space.point_location = [
        (F(0), F(0)),
        (F(4), F(1)),
        (F(1), F(3)),
    ]
    space.line_info = [[0,1],[1,2],[2,0]]
def given_equal_line(line1, line2):
    global space
    line1 = [line_sort(line1)]
    line2 = [line_sort(line2)]
    space.line_eq.data.append([line1, line2])
def given_equal_angle(angle1, angle2):
    global space
    space.angle_eq.data.append([[space.standard_angle(angle1)], [space.standard_angle(angle2)]])
def cpct():
    global space
    for item in space.tri_eq.logic2d():
        for item2 in itertools.combinations(item, 2):
            for item3 in itertools.permutations(list(zip(*item2))):
                angle1, angle2 = (item3[0][0], item3[1][0], item3[2][0]), (item3[0][1], item3[1][1], item3[2][1])
                angle1, angle2 = space.standard_angle(angle1), space.standard_angle(angle2)                
                if angle1 is None or angle2 is None or angle1 == angle2:
                    continue
                space.angle_eq.data.append([[angle1], [angle2]])
            for item3 in zip(list(itertools.combinations(item2[0], 2)), list(itertools.combinations(item2[1], 2))):
                line1, line2 = item3
                line1, line2 = line_sort(line1), line_sort(line2)
                if not space.valid_line(line1) or not space.valid_line(line2) or line1 == line2:
                    continue
                space.line_eq.data.append([[line1], [line2]])
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
def aas_rule(a1, a2, a3, b1, b2, b3):
    global space
    a1,a2,a3,b1,b2,b3 = [[x] for x in [a1,a2,a3,b1,b2,b3]]
    angles = [
        space.standard_angle(a1+a2+a3),
        space.standard_angle(a2+a3+a1),
        space.standard_angle(b1+b2+b3),
        space.standard_angle(b2+b3+b1)
    ]
    if any(x is None for x in angles):
        return False
    side = [
        line_sort(a1+a3),
        line_sort(b1+b3)
    ]
    if not all(space.valid_line(x) for x in side):
        return False
    return (
        space.angle_eq_fx(angles[0], angles[2])
        and
        space.angle_eq_fx(angles[1], angles[3])
        and
        space.line_eq_fx(side[0], side[1])
    )
def rhs_rule(a1,a2,a3,b1,b2,b3):
    global space
    a1,a2,a3,b1,b2,b3 = [[x] for x in [a1,a2,a3,b1,b2,b3]]
    leg = [
        line_sort(a1+a2),
        line_sort(b1+b2)
    ]
    hyp = [
        line_sort(a1+a3),
        line_sort(b1+b3)
    ]
    angle = [
        space.standard_angle(a1+a2+a3),
        space.standard_angle(b1+b2+b3)
    ]
    if None in angle:
        return False
    if not all(space.valid_line(x) for x in leg+hyp):
        return False
    return (
        space.line_eq_fx(leg[0], leg[1])
        and space.line_eq_fx(hyp[0], hyp[1])
        and angle[0] in space.perpendicular_angle
        and angle[1] in space.perpendicular_angle
    )
def sas_rule(a1, a2, a3, b1, b2, b3):
    global space
    a1, a2, a3, b1, b2, b3 = [[item] for item in [a1, a2, a3, b1, b2, b3]]
    line = [
        line_sort(a1 + a2),
        line_sort(b1 + b2),
        line_sort(a2 + a3),
        line_sort(b2 + b3),
    ]
    angle = [space.standard_angle(a1 + a2 + a3), space.standard_angle(b1 + b2 + b3)]
    for item in line:
        if not space.valid_line(item):
            return False
    for item in angle:
        if item is None:
            return False
        
    if space.angle_eq_fx(angle[0], angle[1]):
        if space.line_eq_fx(line[0], line[1]) and space.line_eq_fx(line[2], line[3]):
            return True
    return False
def asa_rule(a1,a2,a3,b1,b2,b3):
    global space
    a1,a2,a3,b1,b2,b3 = [[x] for x in [a1,a2,a3,b1,b2,b3]]
    side = [
        line_sort(a1+a2),
        line_sort(b1+b2)
    ]
    angles = [
        space.standard_angle(a1+a2+a3),
        space.standard_angle(b1+b2+b3),
        space.standard_angle(a2+a1+a3),
        space.standard_angle(b2+b1+b3)
    ]
    if any(x is None for x in angles):
        return False
    if not all(space.valid_line(x) for x in side):
        return False
    return (
        space.angle_eq_fx(angles[0], angles[1])
        and space.angle_eq_fx(angles[2], angles[3])
        and space.line_eq_fx(side[0], side[1])
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
    for item in itertools.product(list1, list2):
        item = list(item[0])+list(item[1])
        for rule in [asa_rule, sss_rule, rhs_rule, sas_rule, aas_rule]:
            out = rule(*item)
            if out:
                space.tri_list.append(tuple(item[:3]))
                space.tri_list.append(tuple(item[-3:]))
                space.tri_list = list(set(space.tri_list))
                space.tri_eq.data.append([[tuple(item[:3])], [tuple(item[-3:])]])
def process():
    global space
    space.calc_line_info()
    space.calc_angle_list()
    space.perpen_angle()
    space.update_arc()
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
    space.line_info.append(list(line))
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
def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])
def dot(u, v):
    return u[0]*v[0] + u[1]*v[1]
def cross(u, v):
    return u[0]*v[1] - u[1]*v[0]
def closest_in_direction_fraction(A, B, points):
    d = sub(B, A)
    dd = dot(d, d)
    candidates = []
    for P in points:
        AP = sub(P, A)
        if cross(AP, d) == 0:
            continue
        num = dot(AP, d)
        if num > 0:
            t = F(num)/F(dd)
            candidates.append((t, P))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]
def point_in_direction_param(A, B, k):
    if isinstance(k, int):
        k = F(k)
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    return (
        A[0] + k * dx,
        A[1] + k * dy
    )
def extend_line(line):
    global space
    p = []
    if isinstance(line, str):
        line = [ord(item)-ord("A") for item in line]
    a, b = space.point_location[line[0]], space.point_location[line[1]]
    for item in itertools.combinations(list(range(len(space.point_location))),2):
        tmp = intersection(space.point_location[item[0]], space.point_location[item[1]], a, b)
        if tmp is None:
            continue
        p.append(tmp)
    out = closest_in_direction_fraction(a, b, p)
    newp = out
    if out is None:
        newp = point_in_direction_param(a, b, 2)
    space.point_location.append(newp)
    join(line_sort((len(space.point_location)-1, line[1])))
def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if yi != yj:
            if (yi > y) != (yj > y):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_intersect:
                    inside = not inside
        j = i
    return inside
def triangle_centroid(a, b, c):
    x = (a[0] + b[0] + c[0]) / 3
    y = (a[1] + b[1] + c[1]) / 3
    return (x, y)
def split_lines(points):
    result = []
    n = len(points)
    for start in range(n):
        for end in range(start + 2, n):
            line = points[start:end+1]
            for i in range(1, len(line)-1):
                left = line[:i+1]
                right = line[i:]
                if len(left) >= 2 and len(right) >= 2:
                    result.append((left, right))
    return result
def generate_equation2():
    global space
    lst = []
    for item in space.line_info:
        for item2 in itertools.combinations(item, 2):
            lst.append(line_sort(list(item2)))
    lst = list(set(lst))    
    for item in space.line_info:
        for item2 in split_lines(item):
            space.line_list += [line_sort(x) for x in item2]
            space.line_list = list(set(space.line_list))
            a = line_sort([item2[0][0], item2[1][-1]])
            b = line_sort([item2[0][0], item2[0][-1]])
            c = line_sort([item2[1][0], item2[1][-1]])
            space.line_eq.data.append([ [a],[b,c] ])

def generate_equation():
    global space
    x = space.graph.all_cycles()
    lst = []
    lst2 = []
    for item in x:
        if set(item) not in lst2:
            lst.append(item)
            lst2.append(set(item))
    lst3 = {}
    for item in lst:
        out = []
        r = []
        for i in range(len(item)):
            if not space.straight_line([item[i-2], item[i-1], item[i]]):
                out.append(item[i-1])
                p = triangle_centroid(*[space.point_location[item2] for item2 in [item[i-2], item[i-1], item[i]]])
                
                if not point_in_polygon(p, [space.point_location[item2] for item2 in item]):
                    r.append(i)
        lst3[tuple(out)] = list(set(r))
        
    for item, reflex in lst3.items():
        if reflex != []:
            continue
        eq = []
        n = 0
        for i in range(len(item)):
            a = space.standard_angle((item[i-2], item[i-1], item[i]))
            eq.append(a)
            n += 1
        n -=2
        n = tree_form("d_180") * tree_form(f"d_{n}")
        eq = tuple(sorted(eq))
        space.angle_val[eq] = n
    for item in space.perpendicular_angle:
        space.angle_val[tuple([item])] = tree_form("d_90")
def access_space():
    global space
    return space
def process2():
    global space
    generate_equation()
    generate_equation2()
    for key, item in space.angle_val.items():
        if len(key) != 1:
            continue
        key = key[0]
        if frac(item) == 90:
            k = space.standard_angle(key)
            if k not in space.perpendicular_angle:
                space.perpendicular_angle.append(k)
    space.angle_eq, space.angle_val = solve_relationship(list(space.angle_list.keys()), space.angle_eq.solve().clean(), space.angle_val)
    space.line_eq, _ = solve_relationship(space.line_list, space.line_eq.solve().clean(), {})
    space.tri_eq, _ = solve_relationship(space.tri_list, space.tri_eq.solve().clean(), {})
def check_angle_value(a):
    global space
    a = space.standard_angle(a)
    if tuple([a]) in space.angle_val.keys():
        return space.angle_val[tuple([a])]
    return None
def vector(a, b):
    global space
    return [space.point_location[b][i] - space.point_location[a][i] for i in range(2)]
def given_line_parallel(a, b):
    global space
    a = space.line_index(a)
    b = space.line_index(b)
    if a == b:
        return
    space.parallel_eq.data.append([[a], [b]])
    space.parallel_list.append(a)
    space.parallel_list.append(b)
    space.parallel_list = list(set(space.parallel_list))
    for item in space.parallel_eq.logic2d():
        for item2 in itertools.combinations(item, 2):
            for item3 in itertools.product(space.line_info[item2[0]], space.line_info[item2[1]]):
                item3 = tuple(item3)
                if space.valid_line(line_sort(list(item3))):
                    item4 = []
                    x = space.line_info[item2[0]].index(item3[0])
                    y = space.line_info[item2[1]].index(item3[1])
                    for i in range(2):
                        item4.append([])
                        if [x,y][i] != 0:
                            item4[-1].append(space.line_info[item2[i]][[x,y][i]-1])
                        if len(space.line_info[item2[i]])-1 != [x,y][i]:
                            item4[-1].append(space.line_info[item2[i]][[x,y][i]+1])
                    
                    for item5 in itertools.product(*item4):
                        p = vector(item3[0], item5[0])
                        q = vector(item3[1], item5[1])
                        if p[0]*q[0] + p[1]*q[1] < 0:
                            tmp = [space.standard_angle([item5[0], item3[0], item3[1]]),space.standard_angle([item5[1], item3[1], item3[0]])]
                            space.angle_eq.data.append([[x] for x in tmp])
def draw_quadrilateral():
    global space
    space = Space()
    space.point_location = [
        (F(0), F(0)),
        (F(4), F(1)),
        (F(3), F(4)),
        (F(1), F(3))
    ]
    space.line_info = [[0,1],[1,2],[2,3],[3,0]]
def given_angle_val(a, val):
    global space
    a = tuple([space.standard_angle(a)])
    space.angle_val[a] = parse(val)
def same_tri_pair(a, b, c, d):
    if set(a) == set(d) or set(b) == set(c):
        c, d = d, c
    if set(a) == set(c) and set(b) == set(d):
        lst = []
        for item in itertools.permutations([0, 1, 2]):
            cnew = tuple([c[item[i]] for i in range(3)])
            dnew = tuple([d[item[i]] for i in range(3)])
            lst.append((cnew, dnew))
        if any((a,b) == item for item in lst):
            return True
    return False
def check_equal_tri(a, b):
    global space
    if isinstance(a, str):
        a = tuple([ord(item)-ord("A") for item in a])
    if isinstance(b, str):
        b = tuple([ord(item)-ord("A") for item in b])
    for item in space.tri_eq.logic2d():
        for item2 in itertools.combinations(item, 2):
            if same_tri_pair(a, b, item2[0], item2[1]):
                return True
    return False
def nothing():
    pass
def god(string):
    lines = [
        line.rstrip()
        for line in string.strip().split("\n")
        if line.strip()
    ]
    block = None
    lst = []
    times = 1
    fx = nothing
    for line in lines:
        text = line.strip()
        if text.endswith(":"):
            block = text[:-1]
            fx()
            if block in ["given", "prove"]:
                fx = process2
            elif block == "construct":
                fx = process
            else:
                fx = nothing
            continue
        parts = text.split()
        if block == "construct":
            if parts[0] == "triangle":
                draw_triangle()
            elif parts[0] == "quadrilateral":
                draw_quadrilateral()
            elif parts[0] == "circle":
                draw_circle()
            elif parts[0] == "arcsplit_minor":
                arc_split(*parts[1], "minor")
            elif parts[0] == "arcsplit_major":
                arc_split(*parts[1], "major")
            elif parts[0] == "perpendicular":
                a,b,c = [parts[1]]+list(parts[2])
                a,b,c = map(lambda x: ord(x)-ord("A"),[a,b,c])
                draw_perpendicular(a, [b,c])
            elif parts[0] == "extend":
                extend_line(parts[1])
            elif parts[0] == "join":
                for x in parts[1:]:
                    join(x)
        elif block == "given":
            if parts[0] == "parallel_line":
                given_line_parallel(parts[1],parts[2])
            elif parts[0] == "line_eq":
                given_equal_line(parts[1],parts[2])
            elif parts[0] == "angle_eq":
                given_equal_angle(parts[1],parts[2])
            elif parts[0] == "angle_val":
                given_angle_val(parts[1], parts[2])
        elif block == "prove":
            if parts[0] == "congruent_triangle":
                prove_congruent_triangle(parts[1],parts[2])
            elif parts[0] == "cpct":
                cpct()
            lst.append(text)
        elif block == "query":
            if parts[0] == "line_eq":
                print(check_equal_line(parts[1],parts[2]))
            elif parts[0] == "angle_eq":
                print(check_equal_angle(parts[1],parts[2]))
            elif parts[0] == "angle_val":
                print(check_angle_value(parts[1]))
            elif parts[0] == "congruent_triangle":
                print(check_equal_tri(parts[1],parts[2]))
        else:
            print("Unknown block:", block)
