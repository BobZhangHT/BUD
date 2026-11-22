/* Batched weighted product-limit factors with the reference operation order. */
#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define BUD_EXPORT __declspec(dllexport)
#else
#define BUD_EXPORT __attribute__((visibility("default")))
#endif

/*
 * weights is row-major. jump columns, tied event totals, and survival
 * columns are contiguous in the replicate dimension. A negative tie slot
 * indicates a single failure whose weight can be read directly.
 */
BUD_EXPORT int bud_km_survival(const double *weights, size_t n_rows,
                                size_t n_subjects, const ptrdiff_t *order,
                                size_t n_jumps, const ptrdiff_t *starts,
                                const ptrdiff_t *first_failure,
                                const ptrdiff_t *tie_slots, const double *tied_totals,
                                double *survival)
{
    enum { BLOCK = 8 };
    for (size_t block = 0; block < n_rows; block += BLOCK) {
        size_t active = n_rows - block < BLOCK ? n_rows - block : BLOCK;
        double risk[BLOCK] = {0.0};
        size_t undone = n_jumps;

        for (size_t step = n_subjects; step > 0; --step) {
            size_t pos = step - 1;
            size_t subject = (size_t)order[pos];
            for (size_t b = 0; b < active; ++b) {
                double weight = weights[(block + b) * n_subjects + subject];
                if (!isfinite(weight) || weight < 0.0)
                    return 1;
                risk[b] += weight;
            }

            if (undone > 0 && pos == (size_t)starts[undone - 1]) {
                size_t jump = --undone;
                ptrdiff_t slot = tie_slots[jump];
                for (size_t b = 0; b < active; ++b) {
                    size_t row = block + b;
                    double failures = slot >= 0
                        ? tied_totals[(size_t)slot * n_rows + row]
                        : weights[row * n_subjects + (size_t)first_failure[jump]];
                    double hazard = risk[b] > 0.0 ? failures / risk[b] : 0.0;
                    if (hazard > 1.0)
                        hazard = 1.0;
                    survival[jump * n_rows + row] = 1.0 - hazard;
                }
            }
        }

        for (size_t b = 0; b < active; ++b) {
            if (!(risk[b] > 0.0))
                return 1;
            risk[b] = 1.0;
        }

        for (size_t jump = 0; jump < n_jumps; ++jump) {
            double *factors = survival + jump * n_rows + block;
            for (size_t b = 0; b < active; ++b) {
                risk[b] *= factors[b];
                factors[b] = risk[b];
            }
        }
    }
    return 0;
}
