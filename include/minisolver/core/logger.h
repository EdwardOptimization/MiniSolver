#pragma once
#include <iostream>

// Log Levels
#define MLOG_LEVEL_NONE  0
#define MLOG_LEVEL_ERROR 1
#define MLOG_LEVEL_WARN  2
#define MLOG_LEVEL_INFO  3
#define MLOG_LEVEL_DEBUG 4

// Default Level: INFO if not defined
#ifndef MINISOLVER_LOG_LEVEL
#define MINISOLVER_LOG_LEVEL MLOG_LEVEL_INFO
#endif

// Log Macros using stream syntax
// Usage: MLOG_INFO("Value: " << x << " iterations");

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_ERROR
    #define MLOG_ERROR(x) do { std::cerr << "\033[31m[ERROR]\033[0m " << x << std::endl; } while(0)
#else
    #define MLOG_ERROR(x) do {} while(0)
#endif

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_WARN
    #define MLOG_WARN(x)  do { std::cout << "\033[33m[WARN] \033[0m " << x << std::endl; } while(0)
#else
    #define MLOG_WARN(x)  do {} while(0)
#endif

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_INFO
    #define MLOG_INFO(x)  do { std::cout << "[INFO]  " << x << std::endl; } while(0)
#else
    #define MLOG_INFO(x)  do {} while(0)
#endif

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_DEBUG
    #define MLOG_DEBUG(x) do { std::cout << "\033[36m[DEBUG]\033[0m " << x << std::endl; } while(0)
#else
    #define MLOG_DEBUG(x) do {} while(0)
#endif

