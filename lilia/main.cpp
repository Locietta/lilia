#include <cstdio>

#include <xmake/version/lilia.hpp>

int main() {
    std::printf("lilia %.*s\n", static_cast<int>(xmake::version::version.size()), xmake::version::version.data());
    return 0;
}
