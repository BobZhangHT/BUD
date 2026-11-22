/*
 * Exact fast path for the phase-II paired sampler.
 *
 * The Python adapter supplies members grouped in the same stable order as
 * NumPy's argsort(inv, kind="stable").  This function finds the two smallest
 * priorities in each group and writes their original row indices in ascending
 * row-index order.  It deliberately returns 1 when a priority tie reaches
 * the second order statistic: NumPy argpartition's tie behaviour is then the
 * authoritative implementation and the adapter falls back to it.
 *
 * Suggested strict builds:
 *   gcc -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -shared ...
 *   cl /std:c11 /O2 /W4 /WX /LD ...
 */
#include <stddef.h>

#if defined(_WIN32)
#define BUD_EXPORT __declspec(dllexport)
#else
#define BUD_EXPORT
#endif

/* Returns 0 on success, 1 for an order-statistic tie, and 2 for bad input. */
BUD_EXPORT int bud_paired_sample(const double *priorities,
                                 const size_t *members,
                                 const size_t *edges,
                                 size_t cells,
                                 size_t *selected) {
    size_t cell;

    if (priorities == NULL || members == NULL || edges == NULL || selected == NULL) {
        return 2;
    }
    for (cell = 0; cell < cells; ++cell) {
        const size_t begin = edges[cell];
        const size_t end = edges[cell + 1];
        size_t first;
        size_t second;
        size_t position;
        double first_value;
        double second_value;

        if (end - begin < 2) {
            return 2;
        }
        first = members[begin];
        second = members[begin + 1];
        first_value = priorities[first];
        second_value = priorities[second];
        if (second_value < first_value) {
            const size_t index_swap = first;
            const double value_swap = first_value;
            first = second;
            first_value = second_value;
            second = index_swap;
            second_value = value_swap;
        }
        for (position = begin + 2; position < end; ++position) {
            const size_t index = members[position];
            const double value = priorities[index];
            if (value < first_value) {
                second = first;
                second_value = first_value;
                first = index;
                first_value = value;
            } else if (value < second_value) {
                second = index;
                second_value = value;
            }
        }
        /* A tie at either selected rank makes argpartition authoritative. */
        for (position = begin; position < end; ++position) {
            const double value = priorities[members[position]];
            if ((value == first_value && first_value == second_value) ||
                (value == second_value && first_value != second_value &&
                 members[position] != second)) {
                return 1;
            }
        }
        if (first < second) {
            selected[2 * cell] = first;
            selected[2 * cell + 1] = second;
        } else {
            selected[2 * cell] = second;
            selected[2 * cell + 1] = first;
        }
    }
    return 0;
}
