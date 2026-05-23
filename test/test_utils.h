#pragma once
#include <cstdio>
#include <cstdlib>

/* ── 간단한 테스트 매크로 (외부 라이브러리 없이) ─────────────── */
static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond) \
    do { \
        if (cond) { \
            printf("  ✓ %s\n", #cond); \
            g_pass++; \
        } else { \
            printf("  ✗ FAIL: %s  (line %d)\n", #cond, __LINE__); \
            g_fail++; \
        } \
    } while(0)

#define TEST_EQ(a, b) \
    do { \
        auto _a = (a); auto _b = (b); \
        if (_a == _b) { \
            printf("  ✓ %s == %s\n", #a, #b); \
            g_pass++; \
        } else { \
            printf("  ✗ FAIL: %s == %s  (%s=%ld, %s=%ld, line %d)\n", \
                   #a, #b, #a, (long)_a, #b, (long)_b, __LINE__); \
            g_fail++; \
        } \
    } while(0)

#define TEST_BEGIN(name) \
    printf("\n=== %s ===\n", name);

#define TEST_SUMMARY() \
    do { \
        printf("\n결과: %d passed, %d failed\n", g_pass, g_fail); \
        return (g_fail > 0) ? 1 : 0; \
    } while(0)
