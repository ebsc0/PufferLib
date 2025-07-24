#include "bandit.h"

#define Env Bandit
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_arms = (int)unpack(kwargs, "num_arms");
    env->horizon = (int)unpack(kwargs, "horizon");
    env->means = calloc(env->num_arms, sizeof(float));
    env->history = calloc(env->horizon, sizeof(float));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    return 0;
}
