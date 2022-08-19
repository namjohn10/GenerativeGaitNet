#include "GLFWApp.h"
#include <pybind11/embed.h>
#include "Environment.h"

int main(int argc, char **argv)
{
    pybind11::scoped_interpreter guard{};
    MASS::Environment *env = new MASS::Environment(true);
    MASS::GLFWApp app(argc, argv);
    app.setEnv(env, argc, argv);
    app.startLoop();
    return 0;
}