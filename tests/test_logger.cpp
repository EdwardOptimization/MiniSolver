#include "minisolver/core/logger.h"
#include <gtest/gtest.h>
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
}
