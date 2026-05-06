#pragma once

#include <sstream>
#include <string>

#if !defined(MINISOLVER_DISABLE_STREAM_LOGGER)
#include <iostream>
#endif

namespace minisolver {

enum class LogLevel { Error = 0, Warn = 1, Info = 2, Debug = 3 };

using LogCallback = void (*)(LogLevel level, const char* message, void* user);

struct LoggerConfig {
    LogCallback callback = nullptr;
    void* user = nullptr;
    bool enable_color = false;
    // When true, log messages with no callback installed are silently dropped
    // instead of being written to stdout/stderr. The embedded compile-time
    // profile MINISOLVER_DISABLE_STREAM_LOGGER defaults this to true so the
    // header keeps no <iostream> dependency on hard real-time targets.
#if defined(MINISOLVER_DISABLE_STREAM_LOGGER)
    bool silent_fallback = true;
#else
    bool silent_fallback = false;
#endif
};

inline LoggerConfig& mutable_logger_config()
{
    static LoggerConfig config;
    return config;
}

inline void set_logger_config(const LoggerConfig& config)
{
    mutable_logger_config() = config;
}

inline LoggerConfig get_logger_config()
{
    return mutable_logger_config();
}

inline const char* log_level_label(LogLevel level)
{
    switch (level) {
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Warn:
        return "WARN";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Debug:
        return "DEBUG";
    default:
        return "UNKNOWN";
    }
}

inline const char* log_level_color(LogLevel level)
{
    switch (level) {
    case LogLevel::Error:
        return "\033[31m";
    case LogLevel::Warn:
        return "\033[33m";
    case LogLevel::Debug:
        return "\033[36m";
    case LogLevel::Info:
    default:
        return "";
    }
}

inline void log_message(LogLevel level, const std::string& message)
{
    const LoggerConfig config = get_logger_config();
    if (config.callback) {
        config.callback(level, message.c_str(), config.user);
        return;
    }

    if (config.silent_fallback) {
        return;
    }

#if defined(MINISOLVER_DISABLE_STREAM_LOGGER)
    // No-stream embedded profile: with no callback and silent_fallback=false the
    // message is still dropped, since <iostream> is not available here.
    (void)level;
    (void)message;
#else
    std::ostream& out = (level == LogLevel::Error) ? std::cerr : std::cout;
    if (config.enable_color && level != LogLevel::Info) {
        out << log_level_color(level) << '[' << log_level_label(level) << "]\033[0m " << message
            << '\n';
        return;
    }

    out << '[' << log_level_label(level) << "] " << message << '\n';
#endif
}

} // namespace minisolver

// Log Levels
#define MLOG_LEVEL_NONE 0
#define MLOG_LEVEL_ERROR 1
#define MLOG_LEVEL_WARN 2
#define MLOG_LEVEL_INFO 3
#define MLOG_LEVEL_DEBUG 4

// Default Level: INFO if not defined
#ifndef MINISOLVER_LOG_LEVEL
#define MINISOLVER_LOG_LEVEL MLOG_LEVEL_INFO
#endif

// Log Macros using stream syntax.
// Usage: MLOG_INFO("Value: " << x << " iterations");

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_ERROR
#define MLOG_ERROR(x)                                                                              \
    do {                                                                                           \
        std::ostringstream minisolver_log_stream__;                                                \
        minisolver_log_stream__ << x;                                                              \
        ::minisolver::log_message(::minisolver::LogLevel::Error, minisolver_log_stream__.str());   \
    } while (0)
#else
#define MLOG_ERROR(x)                                                                              \
    do {                                                                                           \
    } while (0)
#endif

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_WARN
#define MLOG_WARN(x)                                                                               \
    do {                                                                                           \
        std::ostringstream minisolver_log_stream__;                                                \
        minisolver_log_stream__ << x;                                                              \
        ::minisolver::log_message(::minisolver::LogLevel::Warn, minisolver_log_stream__.str());    \
    } while (0)
#else
#define MLOG_WARN(x)                                                                               \
    do {                                                                                           \
    } while (0)
#endif

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_INFO
#define MLOG_INFO(x)                                                                               \
    do {                                                                                           \
        std::ostringstream minisolver_log_stream__;                                                \
        minisolver_log_stream__ << x;                                                              \
        ::minisolver::log_message(::minisolver::LogLevel::Info, minisolver_log_stream__.str());    \
    } while (0)
#else
#define MLOG_INFO(x)                                                                               \
    do {                                                                                           \
    } while (0)
#endif

#if MINISOLVER_LOG_LEVEL >= MLOG_LEVEL_DEBUG
#define MLOG_DEBUG(x)                                                                              \
    do {                                                                                           \
        std::ostringstream minisolver_log_stream__;                                                \
        minisolver_log_stream__ << x;                                                              \
        ::minisolver::log_message(::minisolver::LogLevel::Debug, minisolver_log_stream__.str());   \
    } while (0)
#else
#define MLOG_DEBUG(x)                                                                              \
    do {                                                                                           \
    } while (0)
#endif
