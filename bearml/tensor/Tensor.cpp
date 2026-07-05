// Tensor is now a class template (see Tensor.h). Its former out-of-line
// definitions (print_precision, flatten) are now inline in the header because
// template members need their definitions visible at instantiation. This
// translation unit is intentionally left empty.
#include "Tensor.h"
