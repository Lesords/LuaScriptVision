#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <string>
#include <sys/stat.h>

namespace node {

inline double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                         const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool file_exists(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

inline bool parse_preview_resolution(const std::string& resolution, int* width, int* height) {
    if (!width || !height) {
        return false;
    }

    size_t sep_pos = resolution.find_first_of("xX");
    if (sep_pos == std::string::npos || sep_pos == 0 || sep_pos >= resolution.length() - 1) {
        return false;
    }

    try {
        *width = std::stoi(resolution.substr(0, sep_pos));
        *height = std::stoi(resolution.substr(sep_pos + 1));
    } catch (const std::exception&) {
        return false;
    }

    return *width > 0 && *height > 0;
}

}  // namespace node
