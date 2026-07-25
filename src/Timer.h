#pragma once

#include "pch.h"
#include <chrono>
#include <time.h>

using namespace std::chrono;
/**
 * @brief Timer class for measuring elapsed time
 *
 */
class Timer {
private:
    steady_clock::time_point _start;
    steady_clock::time_point _end;

public:
    double elapsed = 0.0;

public:
    Timer() {
        reset();
    }

    void reset() {
        elapsed = 0.0;
    }

    void tic() {
        _start = steady_clock::now();
    }

    void toc() {
        _end = steady_clock::now();
        elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
    }
};