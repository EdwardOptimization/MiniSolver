#include "minisolver/core/logger.h"
#include <gtest/gtest.h>
#include <iostream>
#include <string>

using namespace minisolver;

namespace {
struct CapturedLog {
    int count = 0;
    LogLevel level = LogLevel::Info;
    std::string message;
};

void capture_log(LogLevel level, const char* message, void* user)
{
    auto* captured = static_cast<CapturedLog*>(user);
    captured->count += 1;
    captured->level = level;
    captured->message = message ? message : "";
}
} // namespace

TEST(LoggerTest, CallbackCapturesStreamStyleWarning)
{
    CapturedLog captured;
    LoggerConfig config;
    config.callback = capture_log;
    config.user = &captured;
    set_logger_config(config);

    MLOG_WARN("api status " << 7);

    set_logger_config(LoggerConfig {});

    EXPECT_EQ(captured.count, 1);
    EXPECT_EQ(captured.level, LogLevel::Warn);
    EXPECT_EQ(captured.message, "api status 7");
}

TEST(LoggerTest, DefaultLoggerConfigKeepsColorDisabled)
{
    set_logger_config(LoggerConfig {});
    const LoggerConfig config = get_logger_config();

    EXPECT_EQ(config.callback, nullptr);
    EXPECT_EQ(config.user, nullptr);
    EXPECT_FALSE(config.enable_color);
#if defined(MINISOLVER_DISABLE_STREAM_LOGGER)
    EXPECT_TRUE(config.silent_fallback);
#else
    EXPECT_FALSE(config.silent_fallback);
#endif
}

TEST(LoggerTest, SilentFallbackDropsMessageWhenCallbackUnset)
{
    // Embedded profile: silent_fallback=true with no callback must not write to
    // stdout or stderr, even at error level. This is the runtime opt-in for the
    // no-stream embedded behavior; MINISOLVER_DISABLE_STREAM_LOGGER provides the
    // equivalent compile-time guarantee that <iostream> stays out of the header.
    LoggerConfig config;
    config.callback = nullptr;
    config.silent_fallback = true;
    set_logger_config(config);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();

    MLOG_ERROR("must be dropped " << 7);
    MLOG_WARN("must be dropped warn");

    const std::string captured_stdout = testing::internal::GetCapturedStdout();
    const std::string captured_stderr = testing::internal::GetCapturedStderr();

    set_logger_config(LoggerConfig {});

    EXPECT_TRUE(captured_stdout.empty()) << "stdout should be empty under silent fallback";
    EXPECT_TRUE(captured_stderr.empty()) << "stderr should be empty under silent fallback";
}

TEST(LoggerTest, SilentFallbackStillRoutesThroughCallback)
{
    CapturedLog captured;
    LoggerConfig config;
    config.callback = capture_log;
    config.user = &captured;
    config.silent_fallback = true;
    set_logger_config(config);

    MLOG_ERROR("payload " << 42);

    set_logger_config(LoggerConfig {});

    EXPECT_EQ(captured.count, 1);
    EXPECT_EQ(captured.level, LogLevel::Error);
    EXPECT_EQ(captured.message, "payload 42");
}

#if defined(MINISOLVER_DISABLE_STREAM_LOGGER)
TEST(LoggerTest, NoStreamProfileDefaultsToSilentFallback)
{
    set_logger_config(LoggerConfig {});
    const LoggerConfig config = get_logger_config();
    EXPECT_TRUE(config.silent_fallback)
        << "MINISOLVER_DISABLE_STREAM_LOGGER must default silent_fallback=true";
}
#endif
