#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <iostream>


namespace mini_llm::runtime {
inline std::vector<int> parseRequest(const std::string& msg) {
    std::vector<int> tokens;
    std::istringstream iss(msg);

    std::string token_str;
    while (iss >> token_str) {
        try {
            tokens.push_back(std::stoi(token_str));
        } catch (...) {
            std::cerr << "[parseRequest] invalid token: " << token_str << "\n";
        }
    }

    return tokens;
}

inline std::string tokensToString(const std::vector<int>& tokens) {
    std::string out;

    for (std::size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) {
            out += " ";
        }
        out += std::to_string(tokens[i]);
    }

    return out;
}
}