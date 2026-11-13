// expcheck — worst relative error of v_expf against libm over the range the engine uses.
#include "../src/kernels.h"
#include <math.h>
#include <stdio.h>
int main(void) {
    double worst = 0; float at = 0;
    for (float x = -90; x < 90; x += 0.00137f) {
        float o[4]; vst1q_f32(o, v_expf(vdupq_n_f32(x)));
        double ref = exp((double)x), err = fabs(o[0] - ref) / ref;
        if (ref > 1e-37 && ref < 1e38 && err > worst) { worst = err; at = x; }
    }
    float o[4]; vst1q_f32(o, v_expf((float32x4_t){ -200, 0, 200, -0.0f }));
    printf("worst relative error %.2e at x = %g\nexp(-200) = %g, exp(0) = %g, exp(200) = %g\n", worst, at, o[0], o[1], o[2]);
    return worst > 1e-6;
}
