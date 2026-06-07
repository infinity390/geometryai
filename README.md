# GeometryAI

## Example: Misc + NCERT class 9 chapter triangles

```python
from geometryai import *
equilateral_triangle = """
construct:
    triangle
given:
    line_eq AB AC
    line_eq BC AC
prove:
    congruent_triangle ABC BAC
    cpct
query:
    angle_val ABC
"""
example_3 = """
construct:
    triangle
    extend AC
    extend BC
    join DE
given:
    parallel_line AB DE
    line_eq AC CD
prove:
    congruent_triangle ECD BCA
    cpct
query:
    congruent_triangle ECD BCA
    line_eq BC CE
"""
exercise_7_1 = """
construct:
    quadrilateral
    join BD
given:
    angle_eq BDA BDC
    line_eq CD AD
prove:
    congruent_triangle BDC BDA
    cpct
query:
    congruent_triangle BDC BDA
    line_eq AB BC
"""
exercise_7_2 = """
construct:
    quadrilateral
    join AC BD
given:
    angle_eq ADC BAD
    line_eq CD AB
prove:
    congruent_triangle ADC DAB
    cpct
query:
    congruent_triangle ADC DAB
    line_eq BD AC
"""
exercise_7_3 = """
construct:
    triangle
    extend AC
    extend BC
    join DE
given:
    angle_val ADE 90
    angle_val BAC 90
    line_eq AB DE
prove:
    congruent_triangle CBA CED
    cpct
query:
    line_eq AC CD
"""
exercise_7_4 = """
construct:
    quadrilateral
    join BD
given:
    parallel_line AB CD
    parallel_line AD BC
prove:
    congruent_triangle CBD ADB
query:
    congruent_triangle CBD ADB
"""
god(equilateral_triangle)
god(example_3)
god(exercise_7_1)
god(exercise_7_2)
god(exercise_7_3)
god(exercise_7_4)
```

### Output

```
60
True
True
True
True
True
True
True
True
```

# Incomplete documentation