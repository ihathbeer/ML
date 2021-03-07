# ML HW2 Q2
# Author: Andrew Nedea
import common
import part1
import part2

d20_train = common.generate_data(20)
d200_train = common.generate_data(200)
d2000_train = common.generate_data(2000)
d10000_validate = common.generate_data(10000)

part1.execute(d10000_validate)
part2.execute(d20_train, d200_train, d2000_train, d10000_validate)

