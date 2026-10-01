#include <math.h>
#include <stddef.h>
void s512_l2_normalize(const float in[512], float out[512]) {
  float sum = 0.0f;
  for (size_t i = 0; i < 512; ++i) sum += in[i] * in[i];
  const float denom = fmaxf(sqrtf(sum), 1e-12f);
  for (size_t i = 0; i < 512; ++i) out[i] = in[i] / denom;
}
