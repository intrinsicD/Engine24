#include "Application.h"

int main() {
    Bcg::Application app;
    app.init(800, 600, "Test BGFX");
    app.run();
    app.cleanup();

    return 0;
}