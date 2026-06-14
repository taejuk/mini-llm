#include "runtime/server.h"

int main() {
    mini_llm::runtime::ServerContext server;
    if(!server.start("0.0.0.0", 8080)) {
        return 1;
    }

    server.run();
    return 0;
}