# GeometryAI

## Example 1: Finding an Angle

```python
from geometryai import *
draw_triangle()
process()
given_equal_angle("ABC", "BCA")
given_equal_angle("CAB", "BCA")
process2()
print(check_angle_value("ABC"))
```

### Output

```
60
```

## Example 2: Parallel lines and congruency

```python
from geometryai import *
draw_triangle()
extend_line("BA")
extend_line("CA")
join("DE")
process()
given_line_parallel("BC", "DE")
given_equal_line("AC", "AE")
prove_congruent_triangle("ACB", "AED")
cpct()
print(check_equal_tri("BAC", "DAE"))
print(check_equal_line("BA", "DA"))
```

### Output

```
True
```

## Example 3: Quadrilaterals and congruency

```python
from geometryai import *
draw_quadrilateral()
join("BD")
process()
given_equal_angle("BDA", "BDC")
given_equal_line("CD", "AD")
prove_congruent_triangle("ADB", "CBD")
cpct()
print(check_equal_line("BC", "BA"))
```

### Output

```
True
```

# Incomplete documentation