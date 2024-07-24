#include "Application.h"

int main() {
    Bcg::Application app;
    app.init(1280, 960, "TestBed");
    app.run();
    app.cleanup();

    return 0;
}