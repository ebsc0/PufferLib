#include "bandit.h"

#define Env Bandit
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_arms = unpack(kwargs, "num_arms");
    env->probs = (float*)calloc(env->num_arms, sizeof(float));
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "total_reward", log->total_reward);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

#define MY_METHODS {NULL, NULL, 0, NULL}
