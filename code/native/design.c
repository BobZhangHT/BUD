/* Rank-one lattice construction, screening, and row-major assignment costs. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#define BUD_EXPORT __declspec(dllexport)
#else
#define BUD_EXPORT __attribute__((visibility("default")))
#endif

static size_t gcd(size_t a, size_t b)
{
    while (b != 0) {
        size_t next = a % b;
        a = b;
        b = next;
    }
    return a;
}

/* A unit's powers first repeat at the return to the identity. */
BUD_EXPORT size_t bud_glp_generators(size_t h, size_t q, size_t *units,
                                     size_t *unit_count, size_t *generators)
{
    size_t count = 0, modulus = h + 1;
    *unit_count = 0;
    for (size_t a = 1; a <= h; ++a) {
        if (gcd(a, modulus) != 1)
            continue;
        units[(*unit_count)++] = a;
        size_t *g = generators + count * q;
        size_t power = 1, j;
        for (j = 0; j < q; ++j) {
            if (j != 0 && power == 1)
                break;
            g[j] = power;
            power = (power * a) % modulus;
        }
        if (j == q)
            ++count;
    }
    return count;
}

BUD_EXPORT void bud_glp_points(size_t h, size_t q, const size_t *generator,
                               double *points)
{
    size_t modulus = h + 1;
    for (size_t k = 0; k < q; ++k) {
        size_t residue = 0;
        for (size_t i = 0; i < h; ++i) {
            residue += generator[k];
            if (residue >= modulus)
                residue -= modulus;
            points[i*q + k] = ((double)residue - 0.5) / (double)h;
        }
    }
}

/* Screen using symmetry. A reference-order rescore resolves close candidates. */
BUD_EXPORT int bud_glp_screen(size_t h, size_t q, size_t count,
                               const size_t *generators, double *scores)
{
    double *term = malloc(h * sizeof(*term));
    double *kernel = malloc(h * h * sizeof(*kernel));
    size_t *row = malloc(h * q * sizeof(*row));
    if (!term || !kernel || !row) {
        free(term);
        free(kernel);
        free(row);
        return 1;
    }
    for (size_t i = 0; i < h; ++i) {
        double x = ((double)i + 0.5) / (double)h;
        double a = fabs(x - 0.5);
        term[i] = 5.0/3.0 - 0.25*a - 0.25*(a*a);
        for (size_t j = 0; j < h; ++j) {
            double y = ((double)j + 0.5) / (double)h;
            double dd = fabs(x-y);
            kernel[i*h + j] = 15.0/8.0 - 0.25*a - 0.25*fabs(y-0.5) - 0.75*dd + 0.5*(dd*dd);
        }
    }
    double constant = pow(19.0/12.0, (double)q);
    double n = (double)h;
    for (size_t c = 0; c < count; ++c) {
        const size_t *g = generators + c*q;
        for (size_t k = 0; k < q; ++k) {
            size_t residue = 0;
            for (size_t i = 0; i < h; ++i) {
                residue += g[k];
                if (residue > h)
                    residue -= h+1;
                row[i*q + k] = residue-1;
            }
        }
        double first = 0.0, diagonal = 0.0, off = 0.0;
        for (size_t i = 0; i < h; ++i) {
            const size_t *left = row + i*q;
            double prod = 1.0, self = 1.0;
            for (size_t k = 0; k < q; ++k) {
                prod *= term[left[k]];
                self *= kernel[left[k]*h + left[k]];
            }
            first += prod;
            diagonal += self;
            for (size_t j = 0; j < i; ++j) {
                const size_t *right = row + j*q;
                prod = 1.0;
                for (size_t k = 0; k < q; ++k)
                    prod *= kernel[left[k]*h + right[k]];
                off += prod;
            }
        }
        scores[c] = constant - (2.0/n)*first + (1.0/(n*n))*(diagonal + 2.0*off);
    }
    free(term);
    free(kernel);
    free(row);
    return 0;
}

/* Write directly in the row-major order consumed by the exact assignment
 * solver. This avoids a column-major gather followed by a full transpose. */
BUD_EXPORT void bud_expand_cost(size_t n, size_t h, const double *distances,
                                const size_t *capacities, double *dense)
{
    for (size_t i = 0; i < n; ++i) {
        double *out = dense + i*n;
        for (size_t j = 0; j < h; ++j) {
            double cost = distances[i*h+j];
            size_t capacity = capacities[j];
            for (size_t k = 0; k < capacity; ++k)
                out[k] = cost;
            out += capacity;
        }
    }
}

/* Linear interpolation in each ordered marginal. The two branches mirror
 * the reference interpolation from each end of the interval. */
BUD_EXPORT void bud_inverse_quantiles(size_t n, size_t h, size_t q,
                                      const double *ordered, const double *design,
                                      double *anchors)
{
    for (size_t i = 0; i < h; ++i) {
        for (size_t k = 0; k < q; ++k) {
            double index = (double)(n-1)*design[i*q+k];
            size_t below = (size_t)index;
            size_t above = below+1 < n ? below+1 : below;
            double fraction = index-(double)below;
            double a = ordered[below*q+k], b = ordered[above*q+k], delta = b-a;
            anchors[i*q+k] = fraction >= 0.5
                ? b-delta*(1.0-fraction)
                : a+delta*fraction;
        }
    }
}
